class modelConfigs:
    """ 
    A class created purely to store some variable values.
    The class is to be used purely for image preprocessing and model training (variable storage) 
    """
    def __init__ (self):
        self.image_paths = [] #inlcude string file paths for each
        self.image_type_number = [] #array for containing each thing
        self.augmentation_probability  = 0.3
        self.epoch_num = 200 #TODO change value to something that makes sense
        self.height = 100 #TODO figure out a height that makes sense
        self.batch_size = 1024 #TODO change to something that makes sense
        self.learning_rate = 1e-4 #may not need thanks to ADAM optimzer

