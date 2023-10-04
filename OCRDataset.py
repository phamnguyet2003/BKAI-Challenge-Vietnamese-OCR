from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torchvision.transforms import ToTensor, Resize, Compose
import torchvision.transforms.functional as F
import os
import numpy as np
from PIL import Image


train_folder_path = '/training_data/new_train/' 
test_folder_path = '/public_test_data/new_public_test/'
label_file_path = '/kaggle/input/handwriting/train_gt.txt/train_gt.txt'
root = '/kaggle/input/handwritten-ocr'

# Encode text to number
def encode_to_num(text, char_list):
    encoded_label = []
    for char in text:
        encoded_label.append(char_list.index(char))
    return encoded_label

class OCRDataset(Dataset):
    def __init__(self, root, max_label_len, train=True, transform=None):
        self.max_label_len = max_label_len
        self.train = train
        self.transform = transform
        if train: # result == list of path of train image file
            dir = root + train_folder_path
            paths = os.listdir(dir)
            # Sort the file names numerically
            paths = sorted(paths, key=lambda x: int(x[10:-4]))
            self.test = paths

            image_files = [os.path.join(dir, path) for path in paths]
            label_file = label_file_path
        else:
            dir = root + test_folder_path
            paths = os.listdir(dir)
            paths = sorted(paths, key=lambda x: int(x[10:-4]))
            image_files = [os.path.join(dir, path) for path in paths]
        
        self.images_path = image_files
        if train:
            self.labels = []
            with open(label_file, encoding='utf-8') as f:
                self.labels = [line.split()[1] for line in f.readlines()]
            char_list= set()
            for label in self.labels:
                char_list.update(set(label))
            self.char_list = sorted(char_list)
            for i in range(len(self.labels)):
                self.labels[i] = encode_to_num(self.labels[i], self.char_list)

    def __len__(self):
        return len(self.images_path)
    def __getitem__(self, idx):      
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = self.labels[idx]
            padded_label = np.squeeze(pad_sequences([label], maxlen=self.max_label_len, padding='post', value = 0))
            return image, padded_label, len(label)
        else:
            return image
        

if __name__ == '__main__':
    transform = Compose([
        Resize((64,128)),
        ToTensor(),
        ])

    ocr = OCRDataset(root = root, max_label_len=32, train=True, transform=transform)
    # print(len(ocr.char_list))
    image, label, length = ocr.__getitem__(447)
    print(image.shape)
    print(label)
    print(length)