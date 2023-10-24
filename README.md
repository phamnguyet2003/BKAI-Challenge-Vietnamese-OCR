# BKAI-Challenge-Vietnamese-OCR
BKAI-NAVER 2023 - Track 2: OCR

### Chủ đề. Cuộc thi Vietnamese Handwritten Text Recognition tập trung vào giải quyết bài toán "Nhận dạng văn bản chữ viết tay Tiếng Việt ".
#### Nhiệm vụ. Cuộc thi tập trung duy nhất vào một nhiệm vụ: nhận dạng chữ viết tay tiếng Việt.

### DATASET: 
The data provided by the organizers consists of three sets as follows:

- Training Data: This is the real labeled dataset used to train the model. It contains 103,000 images, including 51,000 "form" images, 48,000 "wild" images, and 4,000 GAN-generated images.

- Public Test: This is an unlabeled dataset used for preliminary evaluation. It contains 33,000 images, including 17,000 "form" images and 16,000 "wild" images.

- Private Test: This is an unlabeled dataset, and detailed information about it will be disclosed during the final round of the competition.

The input to the model is raw, unlabeled images. The label files are in .txt format. Each line of the label file contains information about the image name and the text label contained within that image in the following format: [image name] [label].
                    IMAGE_NAME   GROUND_TRUTH_TEXT 
                  
      ./kaggle/input/
      ├── handwriting
      │ ├── tensorboard
      │ │ └── tenserboard
      │ ├── train_gt
      │ │ └── train_gt.txt (FILE GIVEN LABEL)
      │ └── last_crnn.pt (MODEL)
      └── handwritten-ocr
      ├── public_test_data
      │ └── new_public_test (TEST DATASET)
      ├── training_data
      │ └── new_train (TRAIN DATASET)
      └── two_characters (UNKNOWN)

[Link of Datasets](https://www.kaggle.com/datasets/phmnhnguyt/handwriting/)

[Link of Datasets 2](https://www.kaggle.com/datasets/ldmkstn/handwritten-ocr/)

### Some Information related to this Repository 
#### 1. Config file
```
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
        self.checkpoint = 'trained_model\last_crnn.pt'
```
#### 2. Dataloader
The provided Python script defines a custom dataset class, OCRDataset, which is used for working with Optical Character Recognition (OCR) data.  It demonstrates how to load an image and its corresponding label for further processing or training. The dataset can be used with PyTorch's DataLoader for efficient data batching during model training.

[OCRDataset.py](https://github.com/phamnguyet2003/BKAI-Challenge-Vietnamese-OCR/blob/main/OCRDataset.py)
#### 3. Model Building
The provided Python script defines a Convolutional Recurrent Neural Network (CRNN) architecture using PyTorch

[model.py](https://github.com/phamnguyet2003/BKAI-Challenge-Vietnamese-OCR/blob/main/model.py)

Components:
- Convolutional Layers (conv1 to conv7): These layers perform feature extraction from input images. They consist of convolutional layers, batch normalization, LeakyReLU activation functions, dropout, and max-pooling operations.
- Fully Connected Layers (fc1): A fully connected layer that reduces the dimensionality of the data.
- Recurrent Layers (rnn1 and rnn2): Two Long Short-Term Memory (LSTM) layers, which are recurrent layers for processing sequential data. They are bidirectional and operate in a batch-first mode.
- Final Fully Connected Layer (fc2): A fully connected layer that produces the model's output.
- Log Softmax (softmax): Applies a logarithmic softmax function to the output.
- He/Kaiming Weight Initialization: Weight initialization using the Kaiming (He) initialization method for convolutional and linear layers.
#### 4. Model Training
Run this code to start to train the model:
```
py train.py
```

### References:
#### [Sample Finished Model](https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb)
#### [Reference of Early Stopping](https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)
