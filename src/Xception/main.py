import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.path_names import ARTICLE_LABELS_PATH, ARTICLE_IMAGES_FOLDER

import tensorflow as tf
from src.utils.evaluate import evaluate_img_model
from src.utils.data_preparation import prepare_img_dataset
import src.utils.functions as fn
from train import train_model, define_model
import argparse
import logging

import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    filename="py_log.log",
    format="%(asctime)s %(levelname)s %(message)s",
)


def main(args: argparse.Namespace) -> None:
    """
    Main function to run Xception Network
    """
    train_dataset, val_dataset, test_dataset = prepare_img_dataset()
    logging.info("Train, validation and test data prepared")

    if args.train:
        model, _ = train_model(train_dataset, val_dataset, args.train_size, args.batch_size)
        logging.info("Model trained")
    elif args.test:
        model = define_model()
        evaluate_img_model(model, test_dataset)
    elif args.test_article:
        model = define_model()
        
        # Prepare article test dataset
        article_test, _ = fn.create_article_dataset(ARTICLE_LABELS_PATH, ARTICLE_IMAGES_FOLDER, image_size=(255, 255))
        
        article_test = article_test.shuffle(512, seed=42)
        article_test = article_test.prefetch(tf.data.experimental.AUTOTUNE)
        
        evaluate_img_model(model, article_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training or inference")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument(
        "--test_article", action="store_true", help="Test the model with unseen data"
    )
    args = parser.parse_args()
    main(args)
