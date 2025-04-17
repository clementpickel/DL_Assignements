import os
import pandas as pd
import numpy as np
import tensorflow as tf

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
random_indices = np.random.permutation(len(nodes_all))
train_indices = random_indices[: int(0.8 * len(random_indices))]
test_indices = random_indices[int(0.8 * len(random_indices)) :]

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

print("Shape of training features:", train_node_features.shape)
print("Shape of training labels:", train_labels.shape)
print("Shape of training masks:", train_mask.shape)

print("Shape of test features:", test_node_features.shape)
print("Shape of test labels:", test_labels.shape)
print("Shape of test masks:", test_mask.shape)
