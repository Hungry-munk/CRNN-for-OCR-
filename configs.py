class modelConfigs:
    """ 
    A class created purely to store some variable values.
    The class is to be used purely for image preprocessing, model training (variable storage)
    and batch generation 
    """
    def __init__ (self):
        self.image_paths = [
            './data/handwritten forms',
            './data/handwritten sentences'
        ] #inlcude string file paths for each
        self.augmentation_probability  = 0.3
        self.data_split = (0.4, 0.6)
        self.epoch_num = 200 #TODO change value to something that makes sense
        self.image_height = 256 #TODO figure out a height that makes sense
        self.batch_size = 1024 #TODO change to something that makes sense
        self.learning_rate = 1e-4 #may not need thanks to ADAM optimzer
        self.letter_to_index = {
            1: 'a',
            2: 'b',
            3: 'c',
            4: 'd',
            5: 'e',
            6: 'f',
            7: 'g',
            8: 'h',
            9: 'j',
            10: 'k',
            11: 'l',
            12: 'm',
            13: 'n',
            14: 'p',
            15: 'r'
        }

