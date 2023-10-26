# Get the sacrebleu scores from the two larger models
from datasets import (combine,
                      load_dataset,)
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

import argparse
import logging
import torch

VERSION = "1.0.0"

def make_logger(filepath):

    # Set up logger
    version_tracking = {"version": "VERSION %s" % VERSION}

    fmt = logging.Formatter(fmt="%(version)s : %(asctime)s : %(message)s",
                            datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger("data_log")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"{filepath}/scores-{VERSION}.log")
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

    return combine.concatenate_datasets(dsets=dsets).shuffle()

def main(args: argparse.ArgumentParser):

    logger = make_logger(args.logging)
    logger.info("===================== RUN =====================")

    # Sync the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"MODEL: {args.model} on {device}")
    pipe = pipeline("translation", args.model, device=device)
    logger.info(f"\t{pipe}")

    # Load the dataset and split into training, testing and validation partitions
    logger.info(f"DATASET(S): {', '.join(args.dataset)}")
    if not args.opus:
        dataset = load_dataset(args.dataset[0], split=args.split if args.split != "" else None) \
            .filter(lambda row: isinstance(row[args.source], str) and isinstance(row[args.target], str))
    else:
        dataset = opus_formatter(args.dataset, lang1=args.source, lang2=args.target, batch_size=args.text_batch_size)
    
    logger.info(f"\t{dataset}")

def add_args(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser if split.py is the main program.
    """

    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="Model location; either on the HugginFace Hub or a local directory.\n \n",
    )

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
        "-tb",
        "--text_batch_size",
        type=int,
        default=32,
        help="batch size for text preprocessing.\n \n",
    )

    parser.add_argument(
        "-me",
        "--metric",
        type=str,
        default="sacrebleu",
        help="Evaluation metric; defaults to sacrebleu.\n \n",
    )

    parser.add_argument(
        "-mk",
        "--metric_keys",
        type=str,
        nargs="*",
        default="score",
        help="Specific keys in the metric dictionary to evaluate the model on.\n \n",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
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
