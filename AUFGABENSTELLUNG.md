---
created: 2026-07-13 23:34
modified: 2026-07-13 23:37
---
# Assignment 3: Music Generation with RNNs

**Einführung in Deep Learning (SoSe26)**

Due date: 17.07.2026, 09:50

## Introduction

In this assignment, you will implement a Recurrent Neural Network (RNN) for music generation. For this you will use the Irish Massive ABC Notation (IrishMAN) dataset, which contains a collection of Irish folk tunes in ABC notation. The goal is to train an RNN to generate new tunes based on the patterns learned from the dataset.

IrishMAN dataset: https://huggingface.co/datasets/sander-wood/irishman

## Tasks

1. **Data Preprocessing**
   Download the IrishMAN dataset and preprocess the ABC notation files to create a suitable input format for the RNN. This includes tokenizing the ABC notation and creating sequences of tokens.

2. **Model Implementation**
   Implement an RNN model (RNN Layer, LSTM Layer) using PyTorch. The model should be able to take sequences of tokens as input and predict the next token in the sequence.

3. **Training**
   Train the RNN on the preprocessed dataset. Experiment with different hyperparameters such as learning rate, batch size, and number of epochs to optimize the model's performance.
  
4. **Music Generation**
   After training, use the RNN to generate new tunes by providing a seed sequence of ABC notation. Implement a method to sample from the model's output to create coherent musical sequences.
  
5. **Evaluation**
   Evaluate the quality of the generated music. Use the following metrics for this:
   - **Top-1 Accuracy** — fraction of sequences where the model predicts the correct next note as the most probable, over all sequences in the test set.
   - **Top-5 Accuracy** — fraction of sequences where the correct note is among the 5 most probable predictions, over all sequences in the test set.
  
   Please use Wandb to visualize the training process and the metrics.
  
## Bonus Points
  
You can earn up to 5 bonus points by implementing one or more of the following extensions:
  
1. **Ablation Study (2 pts)**
   Implement and compare different RNN architectures such as GRU or bidirectional RNNs.
  
2. **Validity Check (2 pts)**
   An analysis that goes beyond top-1 accuracy. Does the model check the syntax of ABC notation? Are bar lines (`|`) set correctly? Does the piece end logically? A small algorithm that evaluates the "grammar" of the generated music.
  
3. **Programming Creativity (1 pt)**
   Implement any additional creative feature or improvement related to the RNN model or music generation process (e.g., audio conversion, particularly impressive visualization, or a cool demo).
  
## Submission Guidelines
  
Submit a ZIP file containing:
  
1. Your Python code for data preparation, model implementation, training, and music generation.
2. Any additional files necessary to run your code (e.g., trained model weights).
3. A one-pager (PDF, font size 12 pt) summarizing your approach, results, and any challenges you faced during the assignment. This includes:
   - A brief description of your data preprocessing steps (only if different from the provided code).
   - An overview of your RNN architecture and hyperparameters.
   - Training curves and evaluation metrics.
   - Examples of generated music in ABC notation format.
  
Be prepared to present your results in the exercise session.