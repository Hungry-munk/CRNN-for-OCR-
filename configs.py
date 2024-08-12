class modelConfigs:
    """ 
    A class created purely to store some variable values.
    The class is to be used purely for image preprocessing, model training (variable storage)
    and batch generation 
    """
    def __init__ (self):
        self.image_paths = [
            './data/handwritten forms',
            './data/handwritten lines'
        ] #inlcude string file paths for each
        self.label_path = "./data/XML Metadata"
        self.augmentation_probability  = 0.3
        self.data_split = (0.4, 0.6)
        self.epoch_num = 200 #TODO change value to something that makes sense
        self.image_height = 512 #TODO figure out a height that makes sense
        self.batch_size = 1024 #TODO change to something that makes sense
        self.learning_rate = 1e-4 #may not need thanks to ADAM optimzer

        #temp attributes
        self.test_batch_size = 100