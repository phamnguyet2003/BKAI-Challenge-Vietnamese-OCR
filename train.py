import os
import torch
import torch.nn as nn
# from OCRDataset import OCRDataset
from torchvision.transforms import ToTensor, Resize, Compose, RandomAffine, ColorJitter
from torch.utils.data import DataLoader
from torch.utils.data import random_split
# from model import CRNN
import itertools
import numpy as np
from argparse import ArgumentParser
# from config import ModelConfigs
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    configs = ModelConfigs()
    root = configs.root
    num_epochs = configs.epochs
    batch_size = configs.batch_size
    train_workers = configs.train_workers
    max_label_len = configs.max_label_len
    height = configs.height
    width = configs.width
    learning_rate = configs.learning_rate
    logging = configs.logging
    trained_models = configs.trained_models
    checkpoint = configs.checkpoint
        
    transform = Compose([
        Resize((height,width)),
        ToTensor(),
         ])
    
    augment_transform= Compose([RandomAffine(
                                            degrees=(-5, 5),
                                            scale=(0.5, 1.05), 
                                            shear=10),
                                ColorJitter(
                                            brightness=0.5, 
                                            contrast=0.5,
                                            saturation=0.5,
                                            hue=0.5)])

    #split train/val dataset
    dataset = OCRDataset(root = root, max_label_len = max_label_len, train=True, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=train_workers,
        drop_last=True,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=train_workers,
        drop_last=True,
        shuffle=True
    )

#     if not os.path.isdir(logging):
#         shutil.rmtree(logging)
#     if not os.path.isdir(trained_models):
#         os.mkdir(trained_models)
    writer = SummaryWriter(logging)

    char_list = dataset.char_list
    model = CRNN(num_classes=len(char_list)+1).to(device)
    criterion = nn.CTCLoss(blank=len(char_list))
    time_steps = max_label_len #time_steps(seq_len) must >= max_label_len but for simplicity, we use time_steps(seq_len) = max_label_len
    output_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = 0
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']  
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"]) 
    else:
        start_epoch = 0  
    num_iters = len(train_dataloader)
    
    ############
    early_stopping = EarlyStopping(patience=3, verbose=True)
    ############
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="green")
        sum_loss = 0
        for iter, (images, padded_labels, label_lenghts) in enumerate(train_dataloader):
            images = augment_transform(images)
            images = images.to(device)
            padded_labels = padded_labels.to(device)
            #forward
            optimizer.zero_grad()
            outputs = model(images)
#Shape:     #output(sequence_length, batch_size, num_classes)
            #padded_labels(batch_size, max_label_len)
            #output_lengths, label_lenghts(batch_size)
            loss_value = criterion(outputs, padded_labels, output_lengths, label_lenghts)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss{:3f}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch*num_iters+iter)
            sum_loss += loss_value
            #backward
            loss_value.backward()  
            optimizer.step()
        
        print(' Avg training loss', sum_loss/11587)
        model.eval()

        for iter, (images, padded_labels, label_lenghts) in enumerate(val_dataloader):
            images = images.to(device)
            padded_labels = padded_labels.to(device)
            with torch.no_grad():
                predictions = model(images)  
                loss_value = criterion(predictions, padded_labels, output_lengths, label_lenghts)
        writer.add_scalar("Val/Loss", loss_value, epoch)
        checkpoint = {
            "epoch": epoch + 1,
            "best_loss" : best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_crnn.pt".format(trained_models))

        if loss_value >= best_loss:
            torch.save(checkpoint, "{}/best_crnn.pt".format(trained_models))
            best_loss = loss_value
        print('Validate', loss_value)
        
        ############
        early_stopping(loss_value, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        ############

