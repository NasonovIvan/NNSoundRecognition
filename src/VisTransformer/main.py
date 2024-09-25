import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.path_names import ARTICLE_LABELS_PATH, ARTICLE_IMAGES_FOLDER, TRAIN_CSV_PATH, SPECTROGRAMS_FOLDER

import argparse
import logging
import torch
from torchvision import transforms
from train import train_model, define_model
from src.utils.data_preparation import prepare_vit_dataset, CustomDataset, ArticleDataset
from src.utils.evaluate import evaluate_vit_model

import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    filename="py_log.log",
    format="%(asctime)s %(levelname)s %(message)s",
)


def main(args):
    """
    Main function to run Visual Transformer Network
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset = CustomDataset(csv_file=TRAIN_CSV_PATH, image_dir=SPECTROGRAMS_FOLDER, transform=transform)
    
    train_dataset, val_dataset, test_dataset = prepare_vit_dataset(full_dataset)
    
    logging.info("Train, validation and test data prepared")

    if args.train:
        model, _ = train_model(train_dataset, val_dataset, args.batch_size)
        logging.info("Model trained")
    elif args.test:
        model = define_model()
        evaluate_vit_model(model, test_dataset)
    elif args.test_article:
        model = define_model()
        
        # Prepare article test dataset
        article_dataset = ArticleDataset(csv_file=ARTICLE_LABELS_PATH,
                                         image_folder=ARTICLE_IMAGES_FOLDER,
                                         transform=transform)
        
        evaluate_vit_model(model, article_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training or inference")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--test_article", action="store_true", help="Test the model with article data")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()
    main(args)