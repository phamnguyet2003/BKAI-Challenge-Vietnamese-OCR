# BKAI-Challenge-Vietnamese-OCR
BKAI-NAVER 2023 - Track 2: OCR

### Chủ đề. Cuộc thi Vietnamese Handwritten Text Recognition tập trung vào giải quyết bài toán "Nhận dạng văn bản chữ viết tay Tiếng Việt ".
#### Nhiệm vụ. Cuộc thi tập trung duy nhất vào một nhiệm vụ: nhận dạng chữ viết tay tiếng Việt.

### DATASET: 
Dữ liệu được cung cấp bởi ban tổ chức gồm 3 tập như sau:
Training data: là tập dữ liệu thật có gán nhãn, dùng để huấn luyện mô hình. Tập này gồm 103000 ảnh (Gồm 51000 ảnh form, 48000 ảnh wild và 4000 ảnh GAN).
Public test: Là tập dữ liệu không nhãn sử dụng để đánh giá vòng sơ loại. Tập này gồm 33000 ảnh. (Gồm 17000 ảnh form và 16000 ảnh wild)
Private test: Là tập dữ liệu không có nhãn. Thông tin chi tiết sẽ công bố tại Vòng chung kết.
Đầu vào cho mô hình là các ảnh thô chưa được gán nhãn. Tệp nhãn là các file định dạng .txt. Mỗi dòng của tệp nhãn chứa thông tin là tên ảnh và nhãn của văn bản chứa trong ảnh đó theo khuôn dạng như sau:
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

### Run this code to start to train the model:
```
py train.py
```

### References:
#### [Sample Finished Model](https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb)
#### [Reference of Early Stopping](https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)
