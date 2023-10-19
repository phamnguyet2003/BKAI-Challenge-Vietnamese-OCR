## BKAI-Challenge-Vietnamese-OCR
[BKAI-NAVER 2023 - Track 2: OCR](https://aihub.vn/competitions/426)

Vietnamese Handwritten Text Recognition, using PyTorch framework
### All of files in this branch is compatible with Kaggle Environment
### DATASET: 

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

### Datasets:
#### [Handwriting](https://www.kaggle.com/datasets/phmnhnguyt/handwriting/)
#### [Handwritten-ocr](https://www.kaggle.com/datasets/ldmkstn/handwritten-ocr/)

### References:
#### [Sample Finished Model](https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb)
#### [Reference of Early Stopping](https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)
