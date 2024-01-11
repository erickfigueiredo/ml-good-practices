import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms_v2


def compile_img_data(partition: str, data_path: str, consider_extensions: set = {'jpg', 'jpeg', 'png'}) -> pd.DataFrame:
    partition_data = []
    for clazz in os.listdir(os.path.join(data_path, partition)):
        for img in os.listdir(os.path.join(data_path, partition, clazz)):
            if img.split('.')[-1] in consider_extensions:
                partition_data.append(
                    {'class': clazz, 'img': os.path.join(data_path, partition, clazz, img)})

    return pd.DataFrame(partition_data)


def plot_class_distribution(df: pd.DataFrame, col_name: str, title: str):
    sns.countplot(x=col_name, data=df, order=sorted(df[col_name].unique()))
    plt.title(f'{title} Distribution')
    plt.xticks(rotation=45)
    plt.show()


class ImageClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: tuple, target: str = None, transform_layers: list = None, shuffle: bool = True) -> None:
        self._data = df
        self._is_training = target is not None

        if self._is_training:
            if shuffle:
                self._shuffle()

            self._class_names = sorted(self._data[target].unique())

        self._transform = self._generate_transforms(img_size, transform_layers)

    @property
    def class_names(self) -> list:
        return self._class_names

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def export_classes_reference(self, save_path='./'):
        if self._is_training:
            pd.DataFrame({'class': self._class_names}).to_csv(os.path.join(save_path, 'reference_classes.csv'), index=False)
        else:
            raise Exception('Your data has no classes reference to export!')

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple:
        img = self._transform(read_image(self._data.loc[idx, 'img'])[:3, :, :])

        if self._is_training:
            label = self._class_names.index(self._data.loc[idx, 'class'])
            return img, label

        return img

    def _shuffle(self) -> None:
        self._data = self._data.sample(frac=1).reset_index(drop=True)

    def _generate_transforms(self, img_size: tuple, transform_layers: list) -> transforms_v2.Compose:
        transforms = [
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize(size=img_size, antialias=True)
        ]

        if transform_layers:
            transforms = list(set(transforms).union(set(transform_layers)))

        return transforms_v2.Compose(transforms)
