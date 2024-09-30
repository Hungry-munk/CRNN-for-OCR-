class Configs:
    """ 
    A class created purely to store some global variable values needed across all files .
    The class is to be used purely for image preprocessing, model training (variable storage)
    and batch generation 
    """
    def __init__ (self):
        self.image_paths = [
            'data/handwritten forms',
            'data/handwritten lines'
        ] #inlcude string file paths for each
        self.label_path = "./data/XML Metadata"
        self.activation_function = 'leaky_relu'
        self.augmentation_probability  = 0.4 
        self.seq_pad_val = -1
        self.epoch_num = 150
        self.image_height = 512
        self.image_max_width = 1024
        self.cv_add_data = 0.075
        self.batch_size = 12
        self.learning_rate = 1e-4
        self.form_height = 3542
        self.form_width = 2479
        self.buffer_size = 1000
        self.epsilon = 1e-7
        # full character vocab found in the labels
        self.char_to_index_map = {
        '<blank>': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
        'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
        't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, 'A': 27, 'B': 28, 'C': 29, 
        'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 
        'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 'U': 47, 'V': 48, 'W': 49, 
        'X': 50, 'Y': 51, 'Z': 52, '0': 53, '1': 54, '2': 55, '3': 56, '4': 57, '5': 58, '6': 59, 
        '7': 60, '8': 61, '9': 62,' ': 63, '.': 64,  ',': 65, '!': 66, '?': 67, ':': 68, ';': 69, 
        '-': 70, "'": 71, '"': 72, '(': 73, ')': 74, '#': 75, '*': 76, '/': 77, '&': 78, '+': 79
        }
        self.index_to_char_map = {
        0: '<blank>', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i',
        10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's',
        20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: 'A', 28: 'B', 29: 'C',
        30: 'D', 31: 'E', 32: 'F', 33: 'G', 34: 'H', 35: 'I', 36: 'J', 37: 'K', 38: 'L', 39: 'M',
        40: 'N', 41: 'O', 42: 'P', 43: 'Q', 44: 'R', 45: 'S', 46: 'T', 47: 'U', 48: 'V', 49: 'W',
        50: 'X', 51: 'Y', 52: 'Z', 53: '0', 54: '1', 55: '2', 56: '3', 57: '4', 58: '5', 59: '6',
        60: '7', 61: '8', 62: '9', 63: ' ', 64: '.', 65: ',', 66: '!', 67: '?', 68: ':', 69: ';',
        70: '-', 71: "'", 72: '"', 73: '(', 74: ')', 75: '#', 76: '*', 77: '/', 78: '&', 79: '+'
        }
        # vocab length
        self.num_classes = len(self.char_to_index_map)