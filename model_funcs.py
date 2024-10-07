import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Conv2D, MaxPooling2D, Dropout, Lambda, Add
from tensorflow.keras.layers import Input, Activation, BatchNormalization
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
from configs import Configs

# get configs
c = Configs()

def character_error_rate(y_true, y_pred):
    # Decode the predicted sequences by taking argmax over time_steps (axis 1)
    y_pred = tf.argmax(y_pred, axis=1)

    # Cast y_pred to int64 to match y_true type
    y_pred = tf.cast(y_pred, tf.int64)
    y_true = tf.cast(y_true, tf.int64)

    # Convert y_true and y_pred to sparse tensors
    y_true_sparse = tf.sparse.from_dense(y_true)
    y_pred_sparse = tf.sparse.from_dense(y_pred)

    # Calculate the edit distance (Levenshtein distance)
    edit_distances = tf.edit_distance(y_true_sparse, y_pred_sparse, normalize=False)

    # Calculate the CER: edit distance / length of true sequence (ignoring blanks)
    true_lengths = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.float32), axis=-1)

    # Avoid division by zero by ensuring true_lengths are at least 1
    true_lengths = tf.maximum(true_lengths, 1)

    cer = edit_distances / true_lengths

    return tf.reduce_mean(cer)

def word_error_rate(y_true, y_pred):
    # Decode the predicted sequences by taking argmax over time_steps (axis 1)
    y_pred = tf.argmax(y_pred, axis=1)

    # Cast y_pred to int64 to match y_true type
    y_pred = tf.cast(y_pred, tf.int64)
    y_true = tf.cast(y_true, tf.int64)

    # Convert y_true and y_pred to sparse tensors
    y_true_sparse = tf.sparse.from_dense(y_true)
    y_pred_sparse = tf.sparse.from_dense(y_pred)

    # Calculate the edit distance (Levenshtein distance)
    edit_distances = tf.edit_distance(y_true_sparse, y_pred_sparse, normalize=False)

    # For WER, assume that the space character (' ') is index 63 (as per your char_to_index_map)
    word_boundaries = tf.cast(tf.equal(y_true, 63), tf.int32)  # Space character separates words
    true_word_lengths = tf.reduce_sum(word_boundaries, axis=-1) + 1  # Adding 1 for the last word

    # Prevent division by zero by ensuring that true_word_lengths are at least 1
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
'''
Not using the loss function
# CTC loss fucntion
class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, name="CTCLoss", blank_index=0, seq_pad_val=-1):
        super(CTCLoss, self).__init__(name=name)
        self.blank_index = blank_index
        self.seq_pad_val = seq_pad_val

    # convert dense tensors into sparse for fater computations
    def dense_to_sparse(self, dense_tensor):
        indices = tf.where(tf.not_equal(dense_tensor, self.seq_pad_val))
        values = tf.gather_nd(dense_tensor, indices)
        shape = tf.shape(dense_tensor, out_type=tf.int64)
        return tf.SparseTensor(indices, values, shape)

    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]

        # Compute input length
        input_length = tf.shape(y_pred)[1]
        input_length = tf.fill([batch_size], input_length)

        # Compute label length
        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, self.seq_pad_val), dtype=tf.int32), axis=1)

        # Convert y_true to sparse tensor
        sparse_labels = self.dense_to_sparse(y_true)
        print()
        # Calculate CTC loss
        ctc_loss = tf.nn.ctc_loss(
            labels=sparse_labels,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=self.blank_index
        )

        # Return average loss for the batch
        return tf.reduce_mean(ctc_loss)
'''
# Custom CTC Loss (modified to handle potential sequence length issues)
class CTCLoss2(tf.keras.losses.Loss):
    def __init__(self, name="CTCLoss", seq_pad_val=-1):
        super().__init__(name=name)
        self.seq_pad_val = seq_pad_val

    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        true_lengths = tf.reduce_sum(tf.cast(tf.not_equal(y_true, self.seq_pad_val), dtype=tf.int32), axis=1)
        input_length = tf.shape(y_pred)[1]
        input_length_tensor = tf.fill([batch_size, 1], input_length)
        label_length = tf.reshape(true_lengths, [-1, 1])
        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length_tensor, label_length)

# take in a sequence and return the string from the label

def ctc_decoder(seqeunce):
    string = ''
    for index in seqeunce.numpy():
        try:
            string += c.index_to_char_map[index]
        except KeyError as e:
            continue
            
    return string

# feature map to seqeunce data
def f_map_to_seq(f_map):
    # Get dynamic shape
    shape = tf.shape(f_map)  # Use dynamic shape to handle None dimensions
    batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    
    # Reshape into (batch_size, timesteps, features)
    return tf.reshape(f_map, (batch_size, width, height * channels))

# function for convelutional bachnorm relu  blocks
# def conv_bn_pool_block(inputs, filters, conv_kernel, conv_padding, block_num, activation, pooling = False, pool_kernel = (2,2)):
#     x = Conv2D(filters, conv_kernel, padding=conv_padding, name=f'conv{block_num}', kernel_initializer='he_normal')(inputs)
#     x = BatchNormalization()(x)
#     x = Activation(activation)(x)
#     if pooling:
#         x = MaxPooling2D(pool_size=pool_kernel, name=f'max{block_num}')(x)
#     return x
def conv_bn_pool_block(inputs, filters, conv_kernel, conv_padding, block_num, activation, pooling=False, pool_kernel=(2,2), use_skip=True):
    x = Conv2D(filters, conv_kernel, padding=conv_padding, name=f'conv{block_num}_1', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    # Add a second convolution layer
    x = Conv2D(filters, conv_kernel, padding=conv_padding, name=f'conv{block_num}_2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    
    # Create skip connection
    if use_skip:
        skip = Conv2D(filters, (1,1), padding='same', name=f'skip{block_num}', kernel_initializer='he_normal')(inputs)
        skip = BatchNormalization()(skip)
        x = Add()([x, skip])
    
    x = Activation(activation)(x)
    
    if pooling:
        x = MaxPooling2D(pool_size=pool_kernel, name=f'max{block_num}')(x)
    
    return x

def build_CRNN_model(input_shape, num_classes, activation='leaky_relu'):
    inputs = Input(shape=input_shape)
    
    # Apply convolutions with skip connections
    f_maps = conv_bn_pool_block(inputs, 64, (5,5), 'same', 1, activation, True, (2,2), use_skip=False)
    f_maps = conv_bn_pool_block(f_maps, 128, (4,4), 'same', 2, activation, True, (2,3))
    f_maps = conv_bn_pool_block(f_maps, 256, (3,3), 'same', 3, activation, False)
    # f_maps = conv_bn_pool_block(f_maps, 512, (3,3), 'same', 4, activation, False)

    # Dropout to help reduce overfitting
    f_maps = Dropout(0.2)(f_maps)

    # CNN to RNN transition: convert the feature maps into sequences
    sequence = Lambda(f_map_to_seq)(f_maps)

    # Dimensionality reduction
    sequence = Dense(128, activation=activation)(sequence)

    # RNN layers (Bidirectional LSTMs)
    sequence = Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='glorot_uniform'))(sequence)
    # sequence = Dropout(0.2)(sequence)
    # sequence = Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='glorot_uniform'))(sequence)

    # Dense layer with softmax activation for classification
    outputs = Dense(num_classes + 1, activation='softmax')(sequence)

    # Create and return model
    model = Model(inputs=inputs, outputs=outputs)
    return model


'''
def build_multi_branch_CRNN(input_shape, num_classes, activation='relu'):
    inputs = Input(shape=input_shape)

    # Shared initial layers
    x = conv_bn_pool_block(inputs, 32, (3, 3), 'same', 1, activation, True, (2, 2))
    x = conv_bn_pool_block(x, 64, (3, 3), 'same', 2, activation, True, (2, 2))

    # Short sequence branch
    short_branch = conv_bn_pool_block(x, 128, (3, 3), 'same', 3, activation, True, (2, 1))
    short_branch = conv_bn_pool_block(short_branch, 128, (3, 3), 'same', 4, activation, False)
    short_branch = Lambda(f_map_to_seq)(short_branch)
    short_branch = Dense(64, activation=activation)(short_branch)  # Dimensionality reduction
    short_branch = Bidirectional(LSTM(128, return_sequences=True))(short_branch)

    # Long sequence branch
    long_branch = conv_bn_pool_block(x, 128, (3, 3), 'same', 5, activation, True, (1, 1))
    long_branch = conv_bn_pool_block(long_branch, 256, (3, 3), 'same', 6, activation, True, (1, 1))
    # long_branch = conv_bn_pool_block(long_branch, 256, (2, 2), 'valid', 7, activation, False)
    long_branch = Lambda(f_map_to_seq)(long_branch)
    long_branch = Dense(128, activation=activation)(long_branch)  # Dimensionality reduction
    long_branch = Bidirectional(LSTM(256, return_sequences=True))(long_branch)

    # Merge branches
    print(long_branch.shape)
    print(short_branch.shape)



    merged = Concatenate()([short_branch, long_branch])
    merged = Dropout(0.5)(merged)
    # merged = Dense(256, activation=activation)(merged)  # Additional dimensionality reduction
    outputs = Dense(num_classes, activation='softmax')(merged)

    model = Model(inputs=inputs, outputs=outputs)
    return model
'''
