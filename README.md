﻿# CRNN for Optical Character Recognition (OCR)

This repository contains the implementation of a Convolutional Recurrent Neural Network (CRNN) model for Optical Character Recognition (OCR). The model is designed to recognize characters from images of text, which can range from individual words to full paragraphs.

The project uses the following key techniques:
Image preprocessing with padding to ensure consistent input dimensions
Normalization and batch normalization to stabilize training
CTC (Connectionist Temporal Classification) loss to handle the alignment of predicted and ground truth sequences
Model Architecture

## Tech Stack:

TensorFlow/Keras (2.10.1): For building, training the CRNN model and building the datapipline.
NumPy: For numerical operations.
Matplotlib: For visualizing the results.

## challenges faced

1. Out of Memory (OOM) Errors on GPU
   One of the early challenges was running into Out of Memory (OOM) errors on the GPU. This occurred due to the large model size and high batch sizes, especially when handling images with longer text sequences.

    Solution:

    To overcome this, I adjusted the model architecture by reducing the number of layers and decreasing the batch size to ensure the model could fit into GPU memory.
    I also experimented with reducing the number of LSTM cells and removing redundant layers, which helped alleviate the memory issues without compromising performance significantly.
    This was a valuable learning experience in optimizing models for specific hardware configurations and understanding the balance between model complexity and available computational resources.

2. Underfitting & Model Architecture Adjustments
   Initially, the model was underfitting the data, which was evident by the large gap between the training loss and validation loss. The model struggled to learn the complexities of the text sequences, especially for images containing longer sentences or paragraphs.

    Solution:

    To address this, I revised the model architecture, reducing the number of pooling layers and tuning kernel sizes in the convolutional layers to extract more relevant features.
    I also modified the LSTM layers, reducing their depth but increasing the sequence length predictions, which significantly improved the model’s ability to capture sequential dependencies in the data.
    These changes allowed the model to generalize better, especially for more complex text sequences, resulting in improved training and validation performance.

3. Model Experimentations and Multi-Branch Architecture
   At one point, I experimented with a multi-branch version of the CRNN architecture to handle longer sequences and make predictions for images containing large amounts of text. The idea was to use multiple parallel branches to extract features from different parts of the image, which could improve the model’s performance on more complex data.

    Result:

    While this approach seemed promising, the multi-branch architecture introduced too many parameters, which increased the computational load significantly and led to impractical training times and memory usage.
    After testing, I decided not to pursue this architecture further, as the overhead in terms of parameters outweighed the benefits for my current dataset and hardware setup.
    This experimentation process taught me a lot about balancing complexity and practicality in model design, and the importance of testing new ideas while being mindful of hardware constraints.

4. Complex Data Preparation & Data Engineering
   The project also required complex data preparation techniques due to the varied nature of the images, with some containing short words and others full paragraphs. Handling this diversity in data required a robust pipeline for:

    Image resizing and padding to ensure consistent input dimensions,
    Label alignment for the CTC loss function, and
    Managing variable sequence lengths efficiently during both training and inference.
    I spent significant time learning about data engineering practices, including how to properly preprocess the images, apply padding, and create a balanced dataset that allows the model to learn effectively from both short and long sequences.

    This aspect of the project greatly improved my understanding of data preparation and the importance of having clean, well-processed data for training deep learning models
    .
