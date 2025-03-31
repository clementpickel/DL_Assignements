# DL_Assignements
Data Science Assignements

# Assignement 1

There is a good tutorial on how to build an image classifier from scratch here: [Keras Image Classification Tutorial](https://keras.io/examples/vision/image_classification_from_scratch/).

**Note:** The cats and dogs dataset is not available anymore from the resource provided in the tutorial, but you can download it from here [Download here].

However, in this tutorial, it is hinted that it can be advantageous to use pre-trained weights, where these come from a model that is trained on a similar dataset (this is called transfer learning). In this assignment, we will test if this is the case and if so, we will test how effective transfer learning is. To achieve this, we will conduct several experiments. You should use the same network architecture and training parameters for all these experiments, and you should record and report the accuracy on the cats and dogs dataset for each epoch in all 4 experiments. To pass this assignment, you should perform all the following bullet points below:

## Experiments

### Experiment 1:
- Follow the tutorial above and train a CNN model to distinguish between cats and dogs.
- Use a learning rate of `0.0001` instead of the one used in the tutorial.
- Train a model (with the same architecture as the model in the tutorial) on the Stanford Dogs dataset: [Stanford Dogs Dataset](https://www.tensorflow.org/datasets/catalog/stanford_dogs). 
- When trained, save this model to a file.

### Experiment 2:
- Load the saved model and replace only the output layer of the model (to align it to the new problem).
- Train and evaluate the model (for 50 epochs) on the cats and dogs dataset.

### Experiment 3:
- Load the saved model and replace the output layer of the model, as well as the first two convolutional layers (keep the weights of all other layers).
- Train and evaluate the model on the cats and dogs dataset.

### Experiment 4:
- Load the saved model and replace the output layer of the model, as well as the two last convolutional layers.
- Train and evaluate the model on the cats and dogs dataset.

## Report Submission
Write a short report (1-2 pages) where you present your results and discuss the reasons behind these. Submit this together with your implementations for all your experiments (either Python scripts or a notebook).

It is okay to conduct the experiments on a smaller subset of these datasets if you have any problem connected to the size of the datasets.

## Quizzes
In order to pass this assignment, you also need to get at least 40% correct on the following quizzes:

- **Q2** - Training Neural Networks
- **Q3** - Convolutional Neural Networks
- **Q4** - Recurrent Neural Networks
- **Q5** - Attention and Transformers
- **Q6** - Explainable Deep Learning
- **Q7** - Deep Learning on Graphs

# Assignment 2

**by He Tan and Florian Westphal**

## Introduction
This assignment builds on what you have learned about Attention Mechanisms and Graph Neural Networks.

The general task is a simplified version of road user trajectory prediction. This task is relevant, for example, for self-driving cars, which need to predict where other traffic participants will be in order to navigate and avoid collisions.

For this type of prediction, it is important to understand how different road users are positioned relative to each other. For example, one would expect a bicyclist to follow the bicycle track unless a pedestrian is standing in the way. One way to encode these relationships between road users is a graph representation.

Given such a graph representation, as well as certain attributes about each road user, such as their current location, type, speed, and heading, you are supposed to predict their future location.

For this task, you are strongly advised to use this tutorial as a starting point: [Graph Attention Network (GAT) Node Classification](https://keras.io/examples/graph/gat_node_classification/).

## Dataset
Unfortunately, there is no publicly available traffic trajectory dataset. Therefore, we are using a pedestrian trajectory dataset recorded in a Japanese mall. Consequently, there are only pedestrians in the provided dataset.

As in the tutorial, the provided dataset consists of two files for each traffic scene:

### `<scene_id>.edges`
- Two columns containing node IDs: `target, source`
- **Note:** The tutorial models directed edges with `source -> target`.
- You can either use undirected edges by modifying the implementation or adding the missing entries to the edges file. 
  - Example: If the line `target, source` exists, add `source, target`.
  - Alternatively, infer which other pedestrians the source node can see in their field of view and only add those. This models that movement decisions are based only on pedestrians in the field of view.

### `<scene_id>.nodes`
- Seven columns with node properties and target values:
  - `node id, current x, current y, previous x, previous y, future x, future y`
- **Notes:**
  - The `previous x` and `previous y` represent the location of the pedestrian 1 second ago. You can use these values directly or infer movement direction and speed.
  - The `future x` and `future y` represent the target values, i.e., the location where the pedestrian will be in 1 second.
  - Some pedestrians do not have `future x` and `future y` coordinates, so you need to filter those for prediction. However, their current and previous locations can still be used when predicting the future locations of other pedestrians.

## Assignment Tasks

### **Task 1**
Adjust the tutorial implementation to perform the given prediction task and conduct a suitable evaluation on a dedicated test set. For example, compute the Euclidean distance between the target point and the predicted point.

### **Task 2**
Perform hyperparameter tuning of the number of attention heads and try a deeper embedding of the node features:
- For the attention heads, evaluate 2 additional settings.
- Instead of the linear transformation of node states suggested in the tutorial, add:
  - One fully connected layer with ReLU activation.
  - One additional fully connected layer.

### **Task 3**
Replace the learned attention mechanism with an attention mechanism based on the Cosine similarity between node vectors.

## Deliverables
- Python script implementing all three versions of the graph attention network:
  1. Basic model
  2. Model with changed embedding
  3. Model with changed attention mechanism
- Short report (1-2 pages) documenting the results of all three tasks.

## Quizzes
In order to pass this assignment, you also need to get at least 40% correct on the following quizzes:

- **Q8** - Generative Models
- **Q9** - Deep Reinforcement Learning
- **Q10** - Self-Supervised Learning
- **Q11** - Computer Vision
- **Q12** - Neurosymbolic Reasoning
