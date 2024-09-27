import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RandomContrast, RandomZoom, Lambda
from tensorflow.keras.utils import pad_sequences
import pathlib as pl
import xml.etree.ElementTree as ET
from configs import Configs 
from html import unescape
import math
# get configs
c = Configs()

def image_resize_normalize(image, target_height, image_max_width):
    # Resize image
    h, w, _ = tf.unstack(tf.shape(image))
    # calcualte aspect ratio to calcualte appriorpate width
    aspect_ratio = tf.cast(w, tf.float32) / tf.cast(h, tf.float32)
    calc_width = tf.cast(target_height * aspect_ratio, tf.int32)
    if  calc_width > image_max_width:
        # an algo for calcluating the maximum height and width of the image within the with the maxmum height and width 
        height = 0 # startin height value
        while (height <= target_height and height * aspect_ratio <= image_max_width):
            height += 1
        #adjust for extra pixel 
        height -= 1
        # calcualte new width that 
        width = tf.cast(height * aspect_ratio, tf.int32)
        
    else:
        width = calc_width
        height = target_height
    # resize the iamge to fit into the set dimensions
    image = tf.image.resize(image, [height, width])
    # add padding to the image if required
    if height < target_height:
        # calcualte remainder height
        remainder_height = target_height - height
        # get top and bottom padding
        top_pad = tf.cast(tf.math.round(remainder_height / 2), dtype=tf.int32)
        bottom_pad = tf.cast(tf.math.floor(remainder_height / 2), dtype= tf.int32)
        # get total height and adjust if rounding issues occured
        calc_height = top_pad + bottom_pad + height
        if calc_height != target_height:
            bottom_pad += target_height - (calc_height)
        paddings = [[top_pad, bottom_pad],[0,0],[0,0]]
        # get RGB constant values
        constant_values = tf.constant(255, dtype=image.dtype )
            # apply padding
        image = tf.pad(image, paddings, constant_values=constant_values)


    # Convert image to float32 and normalize to [0, 1]
    image = tf.cast(image, tf.float32)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

# a function to seperate the forms imges computer text written parts from the hand written parts
# important as label would need to be doubled to train both parts and other training complications
# the name portion on the form images is also needs to be removed as their is not training data on the text
def forms_text_seporator(form_path, HW_bounding_box):
    # read file in based on file path
    image = tf.io.read_file(form_path)
    image = tf.io.decode_image(image, channels=1) #decode image to grayscale
    # configs for dimensions
    # get the width of the image
    right_pixel = tf.shape(image)[1] - 1 #subtract 1 as images are counted starting at 0
    # bouding box in the convention [y1, x1, y2, x2]
    CW_bounding_box = [0, 0, HW_bounding_box[0] , right_pixel] 

    # crop original form image to just handwritten (hence HW) part using correct bounding box
    HW_cropped_image = tf.image.crop_to_bounding_box(
        image, 
        HW_bounding_box[0],
        HW_bounding_box[1],
        HW_bounding_box[2] - HW_bounding_box[0],
        HW_bounding_box[3] - HW_bounding_box[1]
    )
    # crop original form image to just computer wirtten (hence CW) part using correct bounding box 
    CW_cropped_image = tf.image.crop_to_bounding_box(
        image, 
        CW_bounding_box[0],
        CW_bounding_box[1],
        CW_bounding_box[2],
        CW_bounding_box[3]
    )
    return HW_cropped_image, CW_cropped_image

def build_augmentation_model():
    augmentation_model = tf.keras.Sequential([
        RandomContrast(factor=0.25),
        RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='constant', fill_value=255)  ,
        Lambda(lambda image: tf.cast(image, tf.float32))  # Ensure output is float32   
    ])
    return augmentation_model

def random_pad(image, max_padding):
    if isinstance(image, str):
        # If image is a file path, read and decode it
        image = tf.io.read_file(image)
        image = tf.image.decode_image(image, channels=1)
        
    elif isinstance(image, tf.Tensor):
        if image.dtype != tf.uint8:
            image = tf.cast(tf.clip_by_value(image * 255, 0, 255), tf.uint8)
        if len(image.shape) != 3:
            raise ValueError('Input type must be a tensor with 3 dimesnsions ')
    else:
        raise ValueError("Input must be a tensor or a file path string")


    # Generate random padding values
    pad_top = tf.random.uniform([], minval=max_padding // 4, maxval=max_padding, dtype=tf.int32)
    pad_bottom = tf.random.uniform([], minval=max_padding // 4, maxval=max_padding, dtype=tf.int32)
    pad_left = tf.random.uniform([], minval=max_padding // 4, maxval=max_padding, dtype=tf.int32)
    pad_right = tf.random.uniform([], minval=max_padding // 4, maxval=max_padding, dtype=tf.int32)
    
    # Create padding tensor to outline how to pad
    paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    # Pad the image
    padded_image = tf.pad(image, paddings, mode='CONSTANT',  constant_values=255)

    return padded_image

# randomly decide how much to pad form HW or CW images 
def form_pad_val_gen():
    return tf.random.uniform([], minval=100, maxval=250, dtype=tf.int32)
# randomly decide how much to pad line images
def line_pad_val_gen():
    return tf.random.uniform([], minval=20, maxval=70, dtype=tf.int32)

#a function to update the form croping bouding box based on new data from new line
def form_crop_bouding_box_updater(current_bounding_box, line, line_num, total_lin_num):
    # current bounding box is the dimensions of the current word adjusted
    words = line.findall('word')
    for word in words:
        chars = word.findall('cmp')
        for char in chars:
            # first x coord
            x_val = int(char.get('x'))
            if current_bounding_box[1] == 0:
                current_bounding_box[1] = x_val
            elif current_bounding_box[1] > x_val:
                current_bounding_box[1] = x_val
            #  second x coord 
            if current_bounding_box[3] < x_val:
                current_bounding_box[3] = x_val
            # handling y coords cases
            y_val = int(char.get('y'))
            if line_num == 0:
                if current_bounding_box[0] == 0:
                    current_bounding_box[0] = y_val
                elif current_bounding_box[0] > y_val:
                    current_bounding_box[0] = y_val
            elif line_num == total_lin_num:
                if current_bounding_box[2] == 0:
                    current_bounding_box[2] = y_val
                elif current_bounding_box[2] < y_val:
                    current_bounding_box[2] = y_val
    return current_bounding_box
#  a function for padding a batch of images and seqeunces to the same sizes repectively
def same_pad_batch(X, Y):
     # get the widest image standardize image width
        widths = [tf.shape(image)[1] for image in X]
        longest_width = max(widths) #get the longest sequence 

        X_batch = []
        for image in X:
            # get left and right padding
            left_pad = tf.cast(tf.math.round((longest_width - tf.shape(image)[1]) / 2), dtype = tf.int32)
            right_pad = tf.cast(tf.math.floor((longest_width - tf.shape(image)[1]) / 2), dtype = tf.int32)
            calc_width = left_pad + right_pad + image.shape[1]
            # correct for right padding if needed 
            if calc_width != longest_width:
                right_pad += longest_width - calc_width
            # create padding tensor
            paddings = [[0,0], [left_pad, right_pad], [0,0]]
            # get RGB constant padding value
            constant_values = tf.constant(255, dtype=image.dtype )
            # apply padding
            image = tf.pad(image, paddings, constant_values=constant_values)
            X_batch.append(image)

        # pad labels
        # returns numpy array so convert X_batch to numpy array too
        Y_batch = pad_sequences(Y, padding = 'post', value = c.seq_pad_val)
        # return data
        return np.array(X_batch), Y_batch

def batch_generator(X_image_paths, Y_image_path , batch_size , image_target_height, image_max_width, augmentation_probability, cv_add_data = 0.2):
    if cv_add_data <= 0 or cv_add_data >= 1:
        raise ValueError('cv_add_data argument should be a float between 0 and 1')
    # directory containing labels for training data
    label_dir = pl.Path(Y_image_path)
    # get configs
    c = Configs()
    # forms and lines paths
    forms_path = X_image_paths[0]
    lines_path = X_image_paths[1]
    # training example and label data X and Y
    X = []
    Y = []
    # keep track of the number files are being added to the data batch
    batch_length_counter = 0
    # data augmentor
    augmentation_model = build_augmentation_model()

    for XML_path in label_dir.iterdir():
        # get XML root element 
        try:
            root = ET.parse(str(XML_path)).getroot()
        except ET.ParseError as e:
            print(f"Error parsing XML file {XML_path}: {e}")
            print('skipping training example')
        except Exception as e:
            print(f"An unexpected error occurred with {XML_path}: {e}")
            print('skipping training example')
            continue #skip to next training example 
        # a lines in the XML file
        
        all_line_ele = root.find('handwritten-part')
        lines = all_line_ele.findall('line')
        # get bounding boxes for handwritten part 
        # bouding box in the convention [y1, x1, y2, x2]
        form_crop_bounding_box = [0] * 4
        form_full_text = '' # will be added onto this string 
        line_counter = 0
        line_nums = len(lines) - 1 # the number of lines
        # sub foilder for form that contains the lines for that form
        subf_path = root.get('id')

        # for lines
        for line in lines:
            
            line_text = line.get('text')
            sequence = []
            # remove HTML chars and replace with just their  correpsonding chars 
            line_text = unescape(line_text)
            # create a sequence label int's for current line using mapped chars and line text 
            for char in line_text:
                try:
                    sequence.append(c.char_to_index_map[char])
                except KeyError:
                # add new keys if missed for some reason
                    new_length = len(c.char_to_index_map)
                    c.char_to_index_map[char] = new_length
                    print('\nnew char added to dict makes sure to change:', char, "at index:", new_length)
            # append to sequence data as a numpy array with data type of int32
            Y.append(np.array(sequence, dtype=np.int32))

            form_full_text += f' {line_text}'

            # image path in the subfolder
            image_subf_path = line.get('id')
            # sulber folder name (is the first 3 chars)
            subf_name = image_subf_path[:3]
            
            full_line_image_path = f'{lines_path}/{subf_name}/{subf_path}/{image_subf_path}.png'
            #process the image 
            line_image = random_pad(full_line_image_path, line_pad_val_gen())
            # preprocess image to append into X
            line_image = image_resize_normalize(line_image, image_target_height, image_max_width)
            # randomly with a chosen probability augment
            if np.random.rand() <= augmentation_probability:
                line_image = augmentation_model(line_image)
            X.append(line_image)
            batch_length_counter += 1
            
            form_crop_bounding_box = form_crop_bouding_box_updater(
                form_crop_bounding_box,
                line,
                line_counter,
                line_nums
            )
            line_counter += 1
        
        HW_sequence = []
        CW_sequence = []
        # add the extra text found in CW images
        CW_extra_text = f'Sentence Database {subf_path}'
        # create a sequence label int's for current line using mapped chars and line text
        # for main text
        
        for char in form_full_text:
            try:
                HW_sequence.append(c.char_to_index_map[char])
            except:
                # add new keys if missed for some reason
                new_length = len(c.char_to_index_map)
                c.char_to_index_map[char] = new_length
                print('\nnew char added to dict makes sure to change:', char, "at index:", new_length)
        for char in CW_extra_text:
            try:
                CW_sequence.append(c.char_to_index_map[char])
            except:
                # add new keys if missed for some reason
                new_length = len(c.char_to_index_map)
                c.char_to_index_map[char] = new_length
                print('\nnew char added to dict makes sure to change:', char, "at index:", new_length)

        # append to sequence data as a numpy array with data type of int32
        np_sequence = np.array(HW_sequence, dtype=np.int32) 
        CW_np_extra_sequence = np.array(CW_sequence, dtype=np.int32)
        # add sequences to sequence list Y
        Y.append(np_sequence)
        Y.append(np.concatenate((CW_np_extra_sequence, np_sequence)))
        # form image path
        full_form_image_path = f'{forms_path}/{subf_path}.png'
        try:
            # crop the image in 2 parts and return images contain the HW and CW portions
            CW_cropped_form_image, HW_cropped_form_image = forms_text_seporator(
                full_form_image_path, 
                form_crop_bounding_box
            )
            # randomly pad all form image for richer training data
            CW_form_image = random_pad(CW_cropped_form_image, form_pad_val_gen())
            HW_form_image = random_pad(HW_cropped_form_image, form_pad_val_gen())
            # resize and and normalize image
            CW_form_image = image_resize_normalize(CW_form_image, image_target_height, image_max_width)
            HW_form_image = image_resize_normalize(HW_form_image, image_target_height, image_max_width)
            # augment images
            if np.random.rand() <= augmentation_probability:
                CW_form_image = augmentation_model(CW_form_image)
            if np.random.rand() <= augmentation_probability:
                HW_form_image = augmentation_model(HW_form_image)
            # add them to data
            X.append(HW_form_image)
            X.append(CW_form_image)
            # keep track of data
            batch_length_counter += 2
        except:
            print('\n one form image could not be preprocessed and t herfore skipped')
            print('skipping training example')

    # once their is enough processed data yield the data and prepare the next batch after some final processing
        cv_data_size = tf.math.ceil(batch_size * cv_add_data)
        total_batch_size = int(batch_size + cv_data_size)
        while batch_length_counter >= total_batch_size:
            # pad batches to consistent size for tensorflow datasets
            X_train_batch, Y_train_batch = same_pad_batch(X[:batch_size], Y[:batch_size])
            X_CV_batch, Y_CV_batch = same_pad_batch(X[batch_size: total_batch_size], Y[batch_size: total_batch_size])
            # yield data
            yield ((X_train_batch, Y_train_batch) , (X_CV_batch, Y_CV_batch))
            # Remove used batch from data
            X = X[total_batch_size:]
            Y = Y[total_batch_size:]
            batch_length_counter -= total_batch_size
    # After finishing all lines, handle remaining samples
    if len(X) > 0 and len(Y) > 0:
        X_train_batch, Y_train_batch = same_pad_batch(X[:- cv_data_size], Y[:-cv_data_size])  # Use remaining data to yield one last batch
        X_CV_batch, Y_CV_batch = same_pad_batch(X[ : -cv_data_size], Y[ : -cv_data_size])
        yield (X_train_batch, Y_train_batch), (X_CV_batch, Y_CV_batch)  # Or however you want to handle the CV data


# a function to create tensorflow datasets for proper data management during training
def create_datasets(X_image_paths, Y_image_path , batch_size , image_target_height,image_max_width, augmentation_probability = 0.35, cv_add_data = 0.2 ):
    # calcualte amount of cv data
    cv_data_size = int(tf.math.ceil(batch_size * cv_add_data))
    # create  tensorflow dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: batch_generator(X_image_paths, Y_image_path , batch_size , image_target_height, image_max_width, augmentation_probability, cv_add_data), #lambda function to get data batch
        # train batch signiture
        output_signature=(
            (
                tf.TensorSpec(shape=(batch_size, image_target_height, None, 1), dtype=tf.float32), #define image shape
                tf.TensorSpec(shape=(batch_size,None,), dtype=tf.int32) # define seqeunce label shape
            ),
            # CV batch signiture
            (
                tf.TensorSpec(shape=(cv_data_size ,image_target_height, None, 1), dtype=tf.float32), #define image shape
                tf.TensorSpec(shape=(cv_data_size, None,), dtype=tf.int32) # define seqeunce label shape
            )
        )
    )
    return dataset