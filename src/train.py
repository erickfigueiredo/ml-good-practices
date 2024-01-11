import os
import argparse
from argparse import Namespace

import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2 as transforms_v2

from utils.constants import DEVICE
from utils.model import optimize, build_dataloader
from architectures.img.classification import DefaultImageClassificationNet
from img.classification import compile_img_data, ImageClassificationDataset


def main(args: Namespace):
    paths = os.listdir(args.path)
    if 'train' not in paths:
        raise Exception('No "train" folder found in data path.')

    train_df = compile_img_data('train', args.path)
    

    train_dataset = ImageClassificationDataset(train_df, (224, 224),  'class', [transforms_v2.RandomRotation(degrees=(-45, 45)),
                                                                                transforms_v2.ColorJitter(brightness=.5, hue=.3),
                                                                                transforms_v2.RandomHorizontalFlip(p=0.5)])

    model = DefaultImageClassificationNet(len(train_dataset.class_names))
    model = model.to(DEVICE)

    train_loader = build_dataloader(train_dataset, args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if 'val' in paths:
        val_df = compile_img_data('val', args.path)
        val_dataset = ImageClassificationDataset(val_df, (224, 224), 'class')
        val_loader = build_dataloader(val_dataset, args.batch_size, shuffle=False)
        optimize(model, criterion, optimizer, train_loader, val_loader, args.epochs, save_path=args.save_path)
    else:
        optimize(model, criterion, optimizer, train_loader, epochs=args.epochs, save_path=args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--save_path', default='../models/model.pth', help='Path to save the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate to use for training.')
    parser.add_argument('--path', default='../data/cat_vs_dog', help='Path to data containing "train" and "val" (optional) folders.')

    main(parser.parse_args())
