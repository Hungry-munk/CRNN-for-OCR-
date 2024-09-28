import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Conv2D, MaxPooling2D, Dropout, Lambda
from tensorflow.keras.layers import Input, Activation, BatchNormalization
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
from configs import Configs

# get configs
c = Configs()

def character_error_rate(y_true, y_pred):
    # First, decode the predicted sequences (argmax over classes to get index)
    y_pred = tf.argmax(y_pred, axis=-1)
    
    # Convert the dense labels (y_true, y_pred) to sparse tensors for edit distance calculation
    y_true_sparse = tf.cast(tf.sparse.from_dense(y_true), dtype=tf.int32)
    y_pred_sparse = tf.cast(tf.sparse.from_dense(y_pred), dtype=tf.int32)

    # Calculate the edit distance between y_true and y_pred (this is Levenshtein distance)
    edit_distances = tf.edit_distance(y_pred_sparse, y_true_sparse, normalize=False)
    
    # Calculate the CER: edit distance / length of true sequence (ignoring blanks)
    true_lengths = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.float32), axis=-1)
    
    # Avoid division by zero by ensuring true_lengths are at least 1
    true_lengths = tf.maximum(true_lengths, 1)

    cer = edit_distances / true_lengths

    return tf.reduce_mean(cer)

def word_error_rate(y_true, y_pred):
    # Assuming y_pred is already in index form, we skip tf.argmax
    # Convert the dense labels to sparse tensors for edit distance calculation
    y_true_sparse = tf.cast(tf.sparse.from_dense(y_true), dtype=tf.int32)
    y_pred_sparse = tf.cast(tf.sparse.from_dense(y_pred), dtype=tf.int32)

    # Calculate the edit distance between true and predicted sequences (Levenshtein distance)
    edit_distances = tf.edit_distance(y_pred_sparse, y_true_sparse, normalize=False)

    # For WER, we assume that the space character (' ') is index 63, as per the char_to_index_map
    word_boundaries = tf.cast(tf.equal(y_true, 63), tf.int32)  # Space character separates words
    true_word_lengths = tf.reduce_sum(word_boundaries, axis=-1) + 1  # Adding 1 for the last word

    # Prevent division by zero by ensuring that true_word_lengths is at least 1
    true_word_lengths = tf.maximum(true_word_lengths, 1)

    # Compute WER as the edit distance divided by the number of words
    wer = edit_distances / tf.cast(true_word_lengths, tf.float32)

    # Return the average WER over the batch
    return tf.reduce_mean(wer)
# convert dense tesnros into sparse for CTC loss calculations
def dense_to_sparse(dense_tensor, seq_pad_val=-1):
    indices = tf.where(tf.not_equal(dense_tensor, seq_pad_val))
    values = tf.gather_nd(dense_tensor, indices)
    shape = tf.shape(dense_tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)

# CTC loss fucntion
def ctc_loss_lambda_func(y_true, y_pred):
    batch_size = tf.shape(y_true)[0]

    # Compute input length
    input_length = tf.shape(y_pred)[1]
    input_length = tf.fill([batch_size], input_length)
    # make padding value same dtpye as y_true sequnece
    seq_pad_val = tf.constant(c.seq_pad_val, dtype=y_true.dtype)
    # Compute label length
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, seq_pad_val), dtype=tf.int32), axis=1)
    # Convert y_true to sparse tensor for faster operations when calcualtin CTC loss
    sparse_labels = dense_to_sparse(y_true)
    # retrive the losses for the batch\

    ctc_loss = tf.nn.ctc_loss(
        labels=sparse_labels, 
        logits=y_pred, 
        label_length=label_length, 
        logit_length=input_length,
        logits_time_major=False,
        blank_index=0
    )

    # return average loss for the batch plus epsilion to prevent dividie by 0 errors
    return tf.reduce_mean(ctc_loss) + c.epsilon


# feature map to seqeunce data
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

    # Dropout to help reduce overfitting if its happening
    f_maps = Dropout(0.1)(f_maps)

    # CNN to RNN transition: convert the feature maps into sequences
    sequence = Lambda(f_map_to_seq)(f_maps)

    # Dimensionality reduction
    sequence = Dense(256, activation='relu')(sequence)  # Reduce the feature dimensionality

    # RNN layers (Bidirectional LSTMs)
    sequence = Bidirectional(LSTM(512, return_sequences=True, kernel_initializer='glorot_uniform'))(sequence)
    sequence = Dropout(0.15)(sequence)
    sequence = Bidirectional(LSTM(512, return_sequences=True, kernel_initializer='glorot_uniform'))(sequence)

    # Dense layer with softmax activation for classification
    outputs = Dense(num_classes, activation='softmax')(sequence)

    # Create and return model
    model = Model(inputs=inputs, outputs=outputs)
    return model