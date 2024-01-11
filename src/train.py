import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2 as transforms_v2

from utils.constants import DEVICE
from utils.model import optimize, build_dataloader
from architectures.img.classification import DefaultImageClassificationNet
from img.classification import compile_img_data, ImageClassificationDataset


PATH = '../data/cat_vs_dog'

if __name__ == '__main__':
    train_df = compile_img_data('train', PATH)
    val_df = compile_img_data('val', PATH)

    train_dataset = ImageClassificationDataset(train_df, (224, 224),  'class', [transforms_v2.RandomRotation(degrees=(-45, 45)),
                                                                                transforms_v2.ColorJitter(brightness=.5, hue=.3),
                                                                                transforms_v2.RandomHorizontalFlip(p=0.5)])
    val_dataset = ImageClassificationDataset(val_df, (224, 224), 'class')

    model = DefaultImageClassificationNet(len(train_dataset.class_names))
    model = model.to(DEVICE)

    train_loader = build_dataloader(train_dataset, 32, shuffle=True)
    val_loader = build_dataloader(val_dataset, 32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimize(model, criterion, optimizer, train_loader, val_loader, 10, save_path='../models/model.pth')
