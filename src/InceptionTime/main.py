import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.path_names import TRAIN_CSV_PATH, TRAIN_WAV_PATH, TRAIN_WAV_NOISED_PATH

from src.utils.evaluate import evaluate_model
from src.utils.data_preparation import load_data, prepare_datasets
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
    Main function to run InceptionTime Network
    """
    df, path_train, path_train_noised = load_data(
        TRAIN_CSV_PATH, TRAIN_WAV_PATH, TRAIN_WAV_NOISED_PATH
    )
    logging.info("Data loaded")
    if args.train or args.test:
        train_data_hc, val_data_hc, test_data_hc = prepare_datasets(
            df, path_train, fn.augment_data
        )
        logging.info("Train and test data got")
    elif args.train_noise or args.test_noise:
        train_data_hc, val_data_hc, test_data_hc = prepare_datasets(
            df, path_train_noised, fn.augment_data
        )
        logging.info("Train and test data got")

    if args.train:
        model = train_model(train_data_hc, val_data_hc)
        logging.info("Model trained")
    elif args.train_noise:
        model = train_model(train_data_hc, val_data_hc)
        logging.info("Model trained")
    elif args.test or args.test_noise:
        model = define_model(train_data_hc)
        evaluate_model(model, test_data_hc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training or inference")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--train_noise", action="store_true", help="Train the model with noise data"
    )
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument(
        "--test_noise", action="store_true", help="Test the model with noise data"
    )
    args = parser.parse_args()
    main(args)
