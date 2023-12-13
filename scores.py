# Get the sacrebleu scores from the two larger models
from datasets import (combine,
                      Dataset,
                      DatasetDict,
                      load_dataset,
                      load_from_disk,)
from itertools import chain
from pathlib import Path
from tqdm import tqdm
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM)

import argparse
import evaluate
import logging
import torch

VERSION = "1.5.0"

def make_logger(filepath, logging_num):

    # Set up logger
    version_tracking = {"version": "VERSION %s" % VERSION}

    fmt = logging.Formatter(fmt="%(version)s : %(asctime)s : %(message)s",
                            datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger("data_log")
    logger.setLevel(logging.INFO)

    if logging_num == -1:
        handler = logging.FileHandler(f"{filepath}/scores-{VERSION}.log")
    else:
        handler = logging.FileHandler(f"{filepath}/scores-{logging_num}-{VERSION}.log")

    handler.setFormatter(fmt)

    logger.addHandler(handler)
    logger = logging.LoggerAdapter(logger, version_tracking)

    return logger

def make_translate(source, target, tokenizer, model, model_name, device, task=""):

    def translate(examples):

        if model_name == "nllb-200-distilled-600M":

            inputs = tokenizer(examples[source], padding="max_length", max_length=200, truncation=True, return_tensors="pt").input_ids.to(device)
            gen_tokens = model.generate(inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target], max_length=200)

        else:

            inputs = [f"{task}: {line}" for line in examples[source]]
            input_ids = tokenizer(inputs, padding="max_length", max_length=200, truncation=True, return_tensors="pt").input_ids.to(device)
            gen_tokens = model.generate(input_ids)

        examples[model_name] = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        return examples
    
    return translate

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

    # Load the logger to make the pretty output files
    logger = make_logger(args.logging, args.logging_number)
    logger.info("===================== RUN =====================")

    # Sync the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the model
    logger.info(f"MODEL(S): {args.model} on {device}")
    model_name = str(args.model).split("/")[-1]
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, max_length=200).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True if args.fast else False)

    # Load the dataset and split into training, testing and validation partitions
    logger.info(f"DATASET(S): {', '.join(args.dataset)}")
    if not args.local:

        if not args.opus:
            dataset = load_dataset(args.dataset[0], split=args.split if args.split != "" else None) \
                .filter(lambda row: isinstance(row[args.source], str) and isinstance(row[args.target], str))
        else:
            dataset = opus_formatter(args.dataset, lang1=args.source, lang2=args.target, batch_size=args.text_batch_size)

    else:
        dataset = load_from_disk(args.dataset[0])

    if isinstance(dataset, DatasetDict):

        logger.info(f"\tSPLITS: {', '.join(dataset.column_names.keys())}")
        logger.info(f"\tCOLUMNS: {', '.join(list(dataset.column_names.values())[0])}")
        logger.info(f"\tROWS: {'-'.join(str(len(dataset[key])) for key in dataset)}")

    elif isinstance(dataset, Dataset):
        logger.info(f"\tCOLUMNS: {', '.join(dataset.column_names)}")
        logger.info(f"\tROWS: {str(len(dataset))}")

    # Load the metric
    metric = evaluate.load(args.metric)
    logger.info(f"METRIC: {metric.name} using {args.metric_key}")

    translate = make_translate(args.source, args.target_code, tokenizer, model, model_name, device, task=args.task)
    dataset = dataset.map(translate, batched=True, batch_size=args.batch_size)

    if isinstance(dataset, DatasetDict):

        for split in dataset.column_names.keys():

            key = list()

            for example in tqdm(dataset[split]):
                pred, ref = example[model_name], example[args.target]
                key.append(metric.compute(predictions=[pred], references=[[ref]])[args.metric_key])

            dataset[split] = dataset[split].add_column(name=f"{model_name}_{args.metric_key}", column=key)

        logger.info(f"NEW COLUMNS: {', '.join(list(dataset.column_names.values())[0])}")
        logger.info(f"Example from {list(dataset.column_names.keys())[0]}: {dataset[list(dataset.column_names.keys())[0]][0]}")

    elif isinstance(dataset, Dataset):

        key = list()

        for example in tqdm(dataset):
            pred, ref = example[model_name], example[args.target]
            key.append(metric.compute(predictions=[pred], references=[[ref]])[args.metric_key])

        dataset = dataset.add_column(name=f"{model_name}_{args.metric_key}", column=key)

        logger.info(f"NEW COLUMNS: {', '.join(dataset.column_names)}")
        logger.info(f"\nExample: {dataset[0]}")

    dataset.save_to_disk(args.output)
    logger.info(f"SAVE LOCATION: {args.output}")

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
        "-f",
        "--fast",
        type=int,
        default=1,
        help="Uses the fast tokenizer; assumes it exists.\n \n",
    )

    parser.add_argument(
        "-sc",
        "--source_code",
        type=str,
        default=None,
        help="Source language code used by the model if required.\n \n",
    )

    parser.add_argument(
        "-tc",
        "--target_code",
        type=str,
        default=None,
        help="Target language code used by the model if required.\n \n",
    )

    parser.add_argument(
        "-ts",
        "--task",
        type=str,
        default="Translate from English to Spanish",
        help="Task for the model if it requires an instruction.\n \n",
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
        "-lc",
        "--local",
        type=int,
        default=0,
        help="Boolean to check if the dataset is local; defaults to False (0).\n \n",
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
        "-b",
        "--batch_size",
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
        "--metric_key",
        type=str,
        default="score",
        help="Specific key in the metric dictionary to evaluate the model on.\n \n",
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
        "-ln",
        "--logging_number",
        type=int,
        default=-1,
        help="If parallelized, which logging file this instance will write to; avoids race conditions. Defaults to -1 (no parallelization).\n \n",
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
        prog="scores.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Get the sacrebleu scores from the given models.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
