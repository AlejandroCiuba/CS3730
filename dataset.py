# Split a translation dataset into training, validation and testing portions with some cleaning
from datasets import (combine,
                      dataset_dict,
                      load_dataset,)
from pathlib import Path

import argparse
import logging

VERSION = "1.1.0"

def make_logger(filepath):

    # Set up logger
    version_tracking = {"version": "VERSION %s" % VERSION}

    fmt = logging.Formatter(fmt="%(version)s : %(asctime)s : %(message)s",
                            datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger("data_log")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"{filepath}/dataset-{VERSION}.log")
    handler.setFormatter(fmt)

    logger.addHandler(handler)
    logger = logging.LoggerAdapter(logger, version_tracking)

    return logger

def opus_formatter(dataset_list, lang1 = "en", lang2 = "es", batch_size = 128):
    """
    Takes a list of opus datasets and combines them into one that is formatted for the rest of the pipeline.
    """

    def preprocess(lines, **kwargs):

        lines[lang1] = [translation[lang1] for translation in lines["translation"]]
        lines[lang2] = [translation[lang2] for translation in lines["translation"]]
        lines["id"]  = [f"{kwargs['name']}-{id}" for id in lines["id"]]

        return lines

    dsets = [load_dataset(name, lang1=lang1, lang2=lang2, split="train") \
             .map(preprocess, fn_kwargs={"name": name}, batched=True, batch_size=batch_size) \
             .remove_columns("translation") for name in dataset_list]
    
    return combine.concatenate_datasets(dsets=dsets).shuffle(seed=42)

def main(args: argparse.ArgumentParser):

    # Make the logger
    logger = make_logger(args.logging)
    logger.info("===================== RUN =====================")

    # Load the dataset and split into training, testing and validation partitions
    logger.info(f"DATASET(S): {', '.join(args.dataset)}")
    if not args.opus:
        dataset = load_dataset(args.dataset[0], split=args.split if args.split != "" else None) \
            .filter(lambda row: isinstance(row[args.source], str) and isinstance(row[args.target], str))
    else:
        dataset = opus_formatter(args.dataset, lang1=args.source, lang2=args.target, batch_size=args.batch_size)
    
    logger.info(f"\tRAW COLUMNS: {', '.join(dataset.column_names)}")
    logger.info(f"\tRAW ROWS: {len(dataset)}")

    logger.info("REMOVING: Bad and Short Translations")
    dataset = dataset.filter(lambda x: x[args.source] != x[args.target] and len(x[args.source]) >= 10)

    logger.info(f"\tPOST-PROCESSING COLUMNS: {', '.join(dataset.column_names)}")
    logger.info(f"\tPOST-PROCESSING ROWS: {len(dataset)}")

    # Separate the test split into test and validation partitions
    dataset = dataset.train_test_split(train_size=args.train_size)
    valid = dataset["test"].train_test_split(test_size=0.5)
    dataset = dataset_dict.DatasetDict({'train': dataset['train'],
                                        'test': valid['test'],
                                        'valid': valid['train'],}).shuffle(seed=42)
    
    logger.info(f"TRAIN-TEST-VALID SPLIT: {'-'.join([str(len(dataset['train'])), str(len(dataset['test'])), str(len(dataset['valid']))])}")

    dataset.save_to_disk(args.output)

def add_args(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser if split.py is the main program.
    """

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        nargs="+",
        default="hackathon-pln-es/Axolotl-Spanish-Nahuatl",
        help="Datasets; either on the HuggingFace Hub or a local directory. Assumes one dataset if opus is False.\n \n",
    )

    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="train",
        help="Split to load; empty string for the full dataset.\n \n",
    )

    parser.add_argument(
        "-sl",
        "--source",
        type=str,
        default="sp",
        help="Column name of the source language.\n \n",
    )

    parser.add_argument(
        "-tl",
        "--target",
        type=str,
        default="nah",
        help="Column name of the target language.\n \n",
    )

    parser.add_argument(
        "-op",
        "--opus",
        type=int,
        default=0,
        help="If the datasets are opus-based; defaults to 0.\n \n",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="batch size for text preprocessing.\n \n",
    )

    parser.add_argument(
        "-tr",
        "--train_size",
        type=float,
        default=0.7,
        help="Percentage in the train split; test and validation are divided evenly among the remainder.\n \n",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Directory to save the models.\n \n",
    )

    parser.add_argument(
        "-lo",
        "--logging",
        type=Path,
        required=True,
        help="Directory to save logging outputs.\n \n",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="version number",
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="segment.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Outputs a segmented .eaf file according to APLS standards.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
