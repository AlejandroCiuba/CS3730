# Merge the results from the two inferenced models into one dataset
from datasets import (combine,
                      Dataset,
                      DatasetDict,
                      load_from_disk,)
from functools import reduce
from pathlib import Path
from tqdm import tqdm

import argparse
import logging

import pandas as pd
import random as rand

VERSION = "1.2.5"

def make_logger(filepath):

    # Set up logger
    version_tracking = {"version": "VERSION %s" % VERSION}

    fmt = logging.Formatter(fmt="%(version)s : %(asctime)s : %(message)s",
                            datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger("data_log")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"{filepath}/merge-{VERSION}.log")  # THE /.*- SHOULD ALWAYS CHANGE BETWEEN PROGRAMS
    handler.setFormatter(fmt)

    logger.addHandler(handler)
    logger = logging.LoggerAdapter(logger, version_tracking)

    return logger

def main(args: argparse.ArgumentParser):
    
    logger = make_logger(args.logging)
    logger.info("===================== RUN =====================")

    # Load the datasets
    datasets = {name: load_from_disk(name) for name in args.datasets}
    logger.info(f"DATASETS: {', '.join(args.datasets)}")

    for dataset in datasets:
        logger.info(f"COLUMNS {dataset}: {', '.join(datasets[dataset]['train'].column_names)}")

        for split in args.splits:
            datasets[dataset][split] = datasets[dataset][split].sort(column_names=args.identity)

    # ALGORITHM:
    # 1. For each split:
    # 2. Get that split for each dataset in datasets
    # 3. Turn them in pd.DataFrames via batching
    # 4. pd.merge
    # 5. Convert back into HuggingFace Datasets
    # 6. Merge into one HuggingFace DatasetDict

    # 1.
    new_dataset = {split: None for split in args.splits}
    for split in args.splits:

        split_datasets = []  # 2. - 3.
        for frames in tqdm(zip(*[dataset[split].to_pandas(batch_size=128, batched=True) for dataset in datasets.values()])):

            # 4. from https://stackoverflow.com/questions/44327999/how-to-merge-multiple-dataframes
            merged_df = reduce(lambda left, right: pd.merge(left=left, right=right, how="inner"), frames)
            split_datasets.append(Dataset.from_pandas(merged_df))  # 5a.

        new_dataset[split] = combine.concatenate_datasets(split_datasets)  # 5b.

    dataset = DatasetDict(new_dataset)  # 6.
    logger.info(f"COMBINED SPLITS: {dataset.column_names}")
    logger.info(f"COMBINED COLUMNS: {dataset['train'].column_names}")

    # Select a sample training row to print out
    logger.info(f"SAMPLE DATA: {dataset['train'][rand.randint(0, len(dataset['train']))]}")

    # Save to disk at the specified location
    path = args.output / f"{'_'.join(path.split('/')[-1] for path in args.datasets)}"
    logger.info(f"SAVED TO: {path}")

    dataset.save_to_disk(path)

def add_args(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser if split.py is the main program.
    """

    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        nargs="+",
        help="Datasets; assumes a local directory and that they are DatasetDicts, not just Datasets.\n \n",
    )

    parser.add_argument(
        "-s",
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Names of the splits for each dataset, must be the same for both (defaults and assumes \"train\").\n \n",
    )

    parser.add_argument(
        "-id",
        "--identity",
        type=str,
        default="id",
        help="Name of the column to sort on, assumes the ids in that row match between datasets.\n \n",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Directory to save the merged dataset, saved as D1_D2_..._DN.\n \n",
    )


    parser.add_argument(
        "-l",
        "--logging",
        type=Path,
        required=True,
        help="Directory to save logging outputs.\n \n",
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="merge.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Merges two datasets together for the repeat task.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
