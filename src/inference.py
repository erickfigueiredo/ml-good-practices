import os
import argparse
import pandas as pd
from argparse import Namespace

from utils.constants import DEVICE
from utils.model import build_dataloader, load_model, evaluate
from img.classification import ImageClassificationDataset
from architectures.img.classification import DefaultImageClassificationNet


def main(args: Namespace):
    images = []
    for img in os.listdir(args.path):
        if img.split('.')[-1] in {'jpg', 'jpeg', 'png'}:
            images.append(os.path.join(args.path, img))

    test_df = pd.DataFrame({'img': images})
    classes = pd.read_csv(args.reference_classes)

    test_dataset = ImageClassificationDataset(test_df, (224, 224))
    test_loader = build_dataloader(test_dataset, args.batch_size, shuffle=False)

    model = DefaultImageClassificationNet(len(classes))
    model = load_model(model, args.model_path)
    model.to(DEVICE)


    test_df['prediction'] = evaluate(model, test_loader)[0]
    test_df['prediction'] = test_df['prediction'].apply(lambda idx: classes.iloc[idx])
    test_df.to_csv(os.path.join(args.save_path, 'inference_result.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting with a Deep Learning model')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for inference.')
    parser.add_argument('--path', type=str, default='../data/cat_vs_dog/prediction', help='Path to "prediction" folder.')
    parser.add_argument('--model_path', type=str, default='../models/cat_dog_classification.pt', help='Path to the model.')
    parser.add_argument('--reference_classes', type=str, default='./reference_classes.csv', help='Path to reference classes csv.')
    parser.add_argument('--save_path', type=str, default='./', help='Path where the CSV file with the inference results will be saved.')
    
    main(parser.parse_args())
