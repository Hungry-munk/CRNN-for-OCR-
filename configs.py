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
        self.augmentation_probability  = 0.4 
        self.epoch_num = 200 #TODO change value to something that makes sense
        self.image_height = 256
        self.batch_size = 13000 #TODO change to something that makes sense
        self.learning_rate = 1e-4 #may not need thanks to ADAM optimzer
        self.form_height = 3542
        self.form_width = 2479
        self.char_to_index_map = {
        '<blank>': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
        'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 'U': 47, 'V': 48, 'W': 49, 'X': 50, 'Y': 51, 'Z': 52,
        '0': 53, '1': 54, '2': 55, '3': 56, '4': 57, '5': 58, '6': 59, '7': 60, '8': 61, '9': 62,
        ' ': 63, '.': 64, ',': 65, '!': 66, '?': 67, ':': 68, ';': 69, '-': 70, "'": 71, '"': 72, '(': 73, ')': 74, '#': 75, '*' : 76, '/' : 77
        }
        self.num_classes = len(self.char_to_index_map)