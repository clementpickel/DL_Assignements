import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Directory containing .nodes and .edges files
DATASET_PATH = "dataset"

# Retrieve all scene identifiers (without extension)
scene_ids = sorted(set(f.split(".")[0] for f in os.listdir(DATASET_PATH)))

# Lists to hold all nodes and edges from all scenes
all_nodes = []
all_edges = []
offset = 0 # Global offset to ensure unique node IDs across scenes

for scene_id in scene_ids:
    nodes_path = os.path.join(DATASET_PATH, f"{scene_id}.nodes")
    edges_path = os.path.join(DATASET_PATH, f"{scene_id}.edges")
    
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        continue
    
    # Load node data
    nodes_df = pd.read_csv(
        nodes_path, header=None,
        names=["node_id", "current_x", "current_y", "prev_x", "prev_y", "future_x", "future_y"],
        na_values=["_", "NA", "NaN", "nan"]
    )
    
    # Drop rows with missing essential position data
    nodes_df = nodes_df.dropna(subset=["current_x", "current_y", "prev_x", "prev_y"])
    nodes_df[["current_x", "current_y", "prev_x", "prev_y"]] = nodes_df[["current_x", "current_y", "prev_x", "prev_y"]].astype(float)
    
    # Map local node IDs to global ones
    local_to_global = {nid: i + offset for i, nid in enumerate(nodes_df["node_id"])}
    nodes_df["global_id"] = nodes_df["node_id"].map(local_to_global)
    
    # Load edge data and map local IDs to global IDs
    edges_df = pd.read_csv(edges_path, header=None, names=["target", "source"])
    edges_df["source"] = edges_df["source"].map(local_to_global)
    edges_df["target"] = edges_df["target"].map(local_to_global)
    
    # Add reverse edges to make the graph undirected
    reversed_edges = edges_df.rename(columns={"source": "target", "target": "source"})
    full_edges = pd.concat([edges_df, reversed_edges])
    
    # Accumulate nodes and edges
    all_nodes.append(nodes_df)
    all_edges.append(full_edges)

    offset += len(nodes_df)
    
# Merge all nodes and edges into global structures
nodes_all = pd.concat(all_nodes).sort_values("global_id")
edges_all = pd.concat(all_edges).dropna().astype(int)

# Create TensorFlow tensors
node_features = tf.convert_to_tensor(
    nodes_all[["current_x", "current_y", "prev_x", "prev_y"]].to_numpy(),
    dtype=tf.float32
)

labels = tf.convert_to_tensor(
    nodes_all[["future_x", "future_y"]].fillna(0.0).to_numpy(),
    dtype=tf.float32
)

mask = tf.convert_to_tensor(
    ~nodes_all[["future_x", "future_y"]].isna().any(axis=1),
    dtype=tf.bool
)

edges = tf.convert_to_tensor(
    edges_all[["target", "source"]].to_numpy(),
    dtype=tf.int64
)

# Split data into training and test sets (80/20 split)
split_ratio = 0.5
random_indices = np.random.permutation(len(nodes_all))
split_idx = int(split_ratio * len(random_indices))
train_indices = random_indices[: split_idx]
test_indices = random_indices[split_idx :]

# Subset features and labels for training and testing
train_node_features = node_features.numpy()[train_indices]
test_node_features = node_features.numpy()[test_indices]

train_labels = labels.numpy()[train_indices]
test_labels = labels.numpy()[test_indices]

train_mask = mask.numpy()[train_indices]
test_mask = mask.numpy()[test_indices]

# Convert subsets back to TensorFlow tensors
train_node_features = tf.convert_to_tensor(train_node_features, dtype=tf.float32)
test_node_features = tf.convert_to_tensor(test_node_features, dtype=tf.float32)

train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)

train_mask = tf.convert_to_tensor(train_mask, dtype=tf.bool)
test_mask = tf.convert_to_tensor(test_mask, dtype=tf.bool)

# print("Shape of training features:", train_node_features.shape)
# print("Shape of training labels:", train_labels.shape)
# print("Shape of training masks:", train_mask.shape)

# print("Shape of test features:", test_node_features.shape)
# print("Shape of test labels:", test_labels.shape)
# print("Shape of test masks:", test_mask.shape)

class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
    
    def build(self, input_shape):
        # input_shape[0] is (num_nodes, feature_dim)
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
            trainable=True,
        )
        # attention kernel takes concatenated pair [h_i || h_j]
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
            trainable=True,
        )
        super().build(input_shape)
        
    def call(self, inputs):
        node_states, edges = inputs

        # 1) Linear transform
        h = tf.matmul(node_states, self.kernel)  # (N, units)

        # 2) Compute attention scores for each edge
        #    gather [h_target, h_source]
        edge_states = tf.gather(h, edges)            # (E, 2, units)
        edge_states = tf.reshape(
            edge_states, (tf.shape(edges)[0], 2 * self.units)
        )
        scores = tf.nn.leaky_relu(
            tf.matmul(edge_states, self.kernel_attention)
        )  # (E,1)
        scores = tf.squeeze(scores, -1)               # (E,)
        
        # 3) Normalize per target node
        scores_exp = tf.exp(tf.clip_by_value(scores, -2, 2))
        denom = tf.math.unsorted_segment_sum(
            data=scores_exp,
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0]
        )
        # broadcast denom back to each edge
        denom_per_edge = tf.gather(denom, edges[:, 0])
        alpha = scores_exp / (denom_per_edge + tf.keras.backend.epsilon())
        
        # 4) Weighted aggregation of neighbor states
        neigh = tf.gather(h, edges[:, 1])             # (E, units)
        out = tf.math.unsorted_segment_sum(
            data=neigh * alpha[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0]
        )
        return out

class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        # create one GraphAttention per head
        self.attention_heads = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        node_states, edges = inputs
        head_outputs = [head([node_states, edges]) for head in self.attention_heads]
        if self.merge_type == "concat":
            h = tf.concat(head_outputs, axis=-1)  # (N, units * num_heads)
        else:
            h = tf.reduce_mean(tf.stack(head_outputs, axis=-1), axis=-1)  # (N, units)
        return tf.nn.relu(h)

class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        node_features,
        edges,
        hidden_units=32,
        num_heads=4,
        num_layers=2,
        output_dim=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        # fixed graph inputs
        self.node_features = node_features
        self.edges = edges
        # initial linear projection
        self.preprocess = keras.Sequential([
            layers.Dense(hidden_units * num_heads, activation="relu"),
            layers.Dense(hidden_units * num_heads, activation="relu"),
        ])
        # stack of multi‑head GAT layers
        self.gat_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads, merge_type="concat")
            for _ in range(num_layers)
        ]
        # final regression head
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        x, edges = inputs
        x = self.preprocess(x)
        for gat in self.gat_layers:
            x = gat([x, edges]) + x   # residual connection
        return self.output_layer(x)   # (N, 2)

    def train_step(self, data):
        indices, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self([self.node_features, self.edges])
            # gather only the nodes in this batch
            y_pred_batch = tf.gather(y_pred, indices)
            loss = self.compiled_loss(y_true, y_pred_batch)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(y_true, y_pred_batch)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        indices, y_true = data
        y_pred = self([self.node_features, self.edges])
        y_pred_batch = tf.gather(y_pred, indices)
        loss = self.compiled_loss(y_true, y_pred_batch)
        self.compiled_metrics.update_state(y_true, y_pred_batch)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        y_pred = self([self.node_features, self.edges])
        return tf.gather(y_pred, indices)

# Hyperparameters
HIDDEN_UNITS = 32         # GAT head size
NUM_LAYERS = 2            # number of GAT layers
OUTPUT_DIM = 2            # (future_x, future_y)

NUM_EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 1e-3      # more stable for regression
PATIENCE = 10

HEAD_OPTIONS = [2, 4, 8]

# Loss and metric definitions
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_mae",
    min_delta=1e-4,
    patience=PATIENCE,
    restore_best_weights=True
)

histories = {}
mae_results = {}
predictions = {}

for num_heads in HEAD_OPTIONS:
    print(f"\n--- Training model with {num_heads} attention heads ---")
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Create GAT model
    gat_model = GraphAttentionNetwork(
        node_features=node_features,
        edges=edges,
        hidden_units=HIDDEN_UNITS,
        num_heads=num_heads,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM
    )

    # Compile model
    gat_model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=[mae_metric]
    )

    # Train model
    history = gat_model.fit(
        x=train_indices,
        y=train_labels,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping],
        verbose=0,
    )

    # Evaluate on test data
    loss, mae = gat_model.evaluate(
        x=test_indices,
        y=test_labels,
        verbose=0
    )

    # Save the results
    histories[num_heads] = history.history
    mae_results[num_heads] = mae
    predictions[num_heads] = gat_model.predict(x=test_indices)

# print("--" * 38 + f"\nTest MAE: {mae:.4f}")

# Get predicted future positions
test_preds = gat_model.predict(x=test_indices)


# 1. MAE curves on the validation set
plt.figure(figsize=(10, 6))
for num_heads in HEAD_OPTIONS:
    val_mae = histories[num_heads]['val_mae']
    plt.plot(val_mae, label=f"{num_heads} heads")
plt.title("Validation MAE per number of attention heads")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 2. Bar chart of test MAE
plt.figure(figsize=(6, 5))
plt.bar(
    [str(h) + " heads" for h in mae_results.keys()],
    [mae_results[h] for h in mae_results],
    color="skyblue"
)
plt.title("Test MAE per model")
plt.ylabel("MAE")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# 3. Visualization of a few predictions vs ground truth for the best model
best_heads = min(mae_results, key=mae_results.get)
best_preds = predictions[best_heads]

print(f"\nBest model: {best_heads} attention heads (Test MAE = {mae_results[best_heads]:.4f})")

# Display a few predictions
num_examples = 5
for i in range(num_examples):
    pred = best_preds[i]
    true = test_labels[i]
    l2_error = tf.norm(pred - true).numpy()
    true_norm = tf.norm(true).numpy()
    percent_error = (l2_error / (true_norm + 1e-8)) * 100

    print(f"\nExample {i+1}")
    print(f"  Prediction   : ({pred[0]:.2f}, {pred[1]:.2f})")
    print(f"  Ground Truth : ({true[0]:.2f}, {true[1]:.2f})")
    print(f"  L2 Error     : {l2_error:.2f}")
    print(f"  % Error      : {percent_error:.2f}%")