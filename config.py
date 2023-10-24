<<<<<<< HEAD
import os
from datetime import datetime

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.trained_models = 'trained_model'
        self.root = 'data'
        self.height = 64
        self.width = 128
        self.max_label_len = 32
        self.epochs = 300        
        self.batch_size = 8
        self.learning_rate = 0.0001
        self.train_workers = 4
        self.logging = 'tensorboard'
=======
import os
from datetime import datetime

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.trained_models = 'trained_model'
        self.root = 'data'
        self.height = 64
        self.width = 128
        self.max_label_len = 32
        self.epochs = 300        
        self.batch_size = 8
        self.learning_rate = 0.0001
        self.train_workers = 4
        self.logging = 'tensorboard'
>>>>>>> 6128da33d6bd755e23118beb51d85a4ac0d32199
        self.checkpoint = 'trained_model\last_crnn.pt'