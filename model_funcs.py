import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Conv2D, MaxPooling2D, Dropout, Lambda
from tensorflow.keras.layers import Input, Activation, BatchNormalization
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
from Levenshtein import distance as levenshtein_distance
from configs import Configs

# get configs
c = Configs()

def character_error_rate(y_true, y_pred):
    # Assuming y_true and y_pred are already in index form, not one-hot encoded
    y_true = K.get_value(y_true)
    y_pred = K.get_value(y_pred)
    
    cer = []
    for true, pred in zip(y_true, y_pred):
        # Filter out the blank labels (typically 0 for CTC)
        true_str = ''.join([chr(char) for char in true if char != 0])
        pred_str = ''.join([chr(char) for char in pred if char != 0])
        
        # Calculate CER using Levenshtein distance
        edit_distance = levenshtein_distance(true_str, pred_str)
        cer.append(edit_distance / len(true_str) if len(true_str) > 0 else 0)

    return np.mean(cer)

def word_error_rate(y_true, y_pred):
    # Assuming y_true and y_pred are already in index form, not one-hot encoded
    y_true = K.get_value(y_pred)
    y_pred = K.get_value(y_pred)
    
    wer = []
    for true, pred in zip(y_true, y_pred):
        # Decode the predictions and ground truths to strings
        true_str = ''.join([chr(char) for char in true if char != 0])
        pred_str = ''.join([chr(char) for char in pred if char != 0])
        
        # Split into words
        true_words = true_str.split()
        pred_words = pred_str.split()
        
        # Calculate WER using Levenshtein distance
        edit_distance = levenshtein_distance(true_words, pred_words)
        wer.append(edit_distance / len(true_words) if len(true_words) > 0 else 0)

    return np.mean(wer)


# CTC loss function
def ctc_loss_lambda_func(y_true, y_pred):
    input_length = K.ones_like(y_pred[:, 0, 0]) * (K.int_shape(y_pred)[1])
    label_length = K.sum(K.cast(K.not_equal(y_true, c.seq_pad_val), 'int32'), axis=-1)
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def f_map_to_seq(f_map):
    # Get dynamic shape
    shape = tf.shape(f_map)  # Use dynamic shape to handle None dimensions
    batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    
    # Reshape into (batch_size, timesteps, features)
    sequence = tf.reshape(f_map, (batch_size, width, height * channels))
    
    return sequence

def build_CRNN_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    # CNN layers

    f_maps = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)

    f_maps = BatchNormalization()(f_maps)
    f_maps = Activation('relu')(f_maps)
    f_maps = MaxPooling2D(pool_size=(1, 2), name='max1')(f_maps) # maintain vertical information

    f_maps = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(f_maps)

    f_maps = BatchNormalization()(f_maps)
    f_maps = Activation('relu')(f_maps)
    f_maps = MaxPooling2D(pool_size=(1, 2), name='max2')(f_maps)

    f_maps = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(f_maps)
    f_maps = BatchNormalization()(f_maps)
    f_maps = Activation('relu')(f_maps)
    f_maps = MaxPooling2D(pool_size=(1, 2), name='max3')(f_maps)
    
    f_maps = Conv2D(512, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(f_maps)
    f_maps = BatchNormalization()(f_maps)
    f_maps = Activation('relu')(f_maps)
    f_maps = MaxPooling2D(pool_size=(1, 2), name='max4')(f_maps)

    # Dropout to help reduce overfitting
    f_maps = Dropout(0.3)(f_maps)

    # CNN to RNN transition: convert the feature maps into sequences
    sequence = Lambda(f_map_to_seq)(f_maps)

    # RNN layers (Bidirectional LSTMs)
    sequence = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='glorot_uniform'))(sequence)
    sequence = Dropout(0.3)(sequence)
    sequence = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='glorot_uniform'))(sequence)

    # Dense layer with softmax activation for classification
    outputs = Dense(num_classes, activation='softmax')(sequence)

    # Create and return model
    model = Model(inputs=inputs, outputs=outputs)
    return model