import torch
from mltu.utils.text_utils import ctc_decoder
from model import CRNN
from torch.utils.data import DataLoader
from OCRDataset import OCRDataset
from torchvision.transforms import ToTensor,Compose,Resize
from config import ModelConfigs
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    configs = ModelConfigs()
    root = configs.root
    batch_size = configs.batch_size
    train_workers = configs.train_workers
    max_label_len = configs.max_label_len
    height = configs.height
    width = configs.width
    checkpoint = configs.checkpoint

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")   
    model = CRNN(num_classes=188).to(device)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model"])

    # Load test data loader
    transform = Compose([
        Resize((height,width)),
        ToTensor(),
        ])

    test_dataset = OCRDataset(root = root, max_label_len=max_label_len, train=False, transform=transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=train_workers,
        drop_last=False,
        shuffle=False
    )

    char_list = OCRDataset(root = root, max_label_len=max_label_len, train=True, transform=transform).char_list
    model.eval()
    image_files = test_dataset.images_path
    image_files = [image_file[21:] for image_file in image_files]

    with open('prediction.txt', 'w', encoding='utf-8') as f:
        for i, images in enumerate(test_dataloader):
            images = images.to(device)
            outputs = model(images)
            
            preds = ctc_decoder(
                outputs.cpu().permute(1, 0, 2).detach().numpy(), 
                char_list
            )
            for j in range(len(preds)):
                predicted_result = f"'{preds[j]}'"

                f.write(f"{image_files[i * len(preds) + j]} {predicted_result}\n")

