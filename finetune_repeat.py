# Finetune model with our approach
from datasets import (combine,
                      DatasetDict,
                      load_dataset,
                      load_from_disk,)
from pathlib import Path
from transformers import (AutoModelForSeq2SeqLM, 
                          AutoTokenizer, 
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          pipeline,)
from tqdm import tqdm

import argparse
import evaluate
import logging
import os
import tensorboard
import torch

import numpy as np
import random as rand

VERSION = "1.1.0"

def make_logger(filepath, mixture):

    # Set up logger
    version_tracking = {"version": "VERSION %s" % VERSION}

    fmt = logging.Formatter(fmt="%(version)s : %(asctime)s : %(message)s",
                            datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger("data_log")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"{filepath}/repeat-{mixture}-{VERSION}.log")
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

def make_preprocess_mt(tokenizer, task = "", source = "", target = "", device = "cuda"):
    """
    Returns a preprocessing function using the given tokenizer.
    """

    def preprocess_mt(examples):
        """
        Tokenizes the inputs and targets to feed to the model.
        """

        padding = "max_length"
        max_length = 200

        inputs = [f"{task}: " + str(i) for i in examples[source]]
        targets = [str(t) for t in examples[target]]

        # Memory reqs grow quadratically with input size, stops at max_length
        tokens = tokenizer(text=inputs, text_target=targets, max_length=max_length, padding=padding, truncation=True, return_tensors="pt").to(device)
        return tokens

    return preprocess_mt

def make_preprocess_rt(tokenizer, translations = "", keys = "", source = "en", device = "cuda"):
    """
    Returns a preprocessing function using the given tokenizer.
    """

    TASK = "Repeat the best translation"

    def preprocess_rt(examples):
        """
        Tokenizes the inputs and targets to feed to the model.
        """

        padding = "max_length"
        max_length = 200

        inputs = []
        for trans in zip(*[examples[col] for col in translations], examples[source]):

            trans = list(trans)
            src = trans.pop(-1)  # Source text is always the last element, remove it to not mix it up
            rand.shuffle(trans)  # Mix up translations to try to avoid bias in the decision

            inputs.append("\n".join([f"{TASK}:", *trans, src]))

        targets = []
        for batch in zip(*[examples[col] for col in translations], *[examples[key] for key in keys]):

            trans, ks = batch[0: len(translations)], batch[len(translations):]  # Separate the translations and keys
            best = ks.index(max(ks))  # The index of the best score should also be the index of the best translation

            targets.append(trans[best])

        # Memory reqs grow quadratically with input size, stops at max_length
        tokens = tokenizer(text=inputs, text_target=targets, max_length=max_length, padding=padding, truncation=True, return_tensors="pt").to(device)
        return tokens

    return preprocess_rt

def make_compute_metrics(tokenizer, metric_name="bleu", metric_keys=[]):

    metric = evaluate.load(metric_name)

    def compute_metrics(eval_pred):

        preds, labels = eval_pred

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        results = metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {f"{metric_name}_{k}" : v for k, v in results.items() if k in metric_keys and len(metric_keys) != 0}
    
    return compute_metrics

def make_trainer(model, tokenizer, dataset,
                 learning_rate = 4e-5, epochs = 1, batch_size = 8, 
                 save_at = 0.5, output = "models", metric_name = "bleu", metric_keys=["bleu"]):
    """
    Creates a trainer for the model.
    """

    save_steps = int((len(dataset["train"]) // batch_size) * save_at) if save_at != 1 else "epoch"

    args = Seq2SeqTrainingArguments(
            output_dir=output,
            evaluation_strategy="steps" if isinstance(save_steps, (int)) else save_steps,
            eval_steps=save_steps if isinstance(save_steps, (int)) else 1,
            logging_strategy="steps",
            logging_steps=100,    # Dirty fix, but I need to get this done
            save_strategy="steps" if isinstance(save_steps, (int)) else save_steps,
            save_steps=save_steps if isinstance(save_steps, (int)) else 1,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=epochs,
            predict_with_generate=True,
            bf16=True,
            load_best_model_at_end=True,
            metric_for_best_model=f"{metric_name}_{metric_keys[0]}",
            report_to="tensorboard",)

    compute_metrics = make_compute_metrics(tokenizer=tokenizer, metric_name=metric_name, metric_keys=metric_keys)

    dc = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    return Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            data_collator=dc,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,)

def preview_translation(model, tokenizer, dataset, task = "", source = "", num_examples = 5, **kwargs):
    
    pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=kwargs["device"])

    for i, row in tqdm(enumerate(dataset)):
        
        trans = pipe(task + ": " + row[source])

        yield (i, row, trans)

        if i == num_examples:
            break

def main(args: argparse.ArgumentParser):

    # Logger setup
    logger = make_logger(args.logging, args.task_mixture)
    logger.info("===================== RUN =====================")

    # Sync the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and tokenizer; we use a sentencepiece-based tokenizer, so we disable fast-tokenization
    logger.info(f"MODEL: {args.model} on {device}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, max_length=256).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Load the preprocessing functions
    logger.info(f"MACHINE TRANSLATION TASK: {args.task}")
    preprocess_mt = make_preprocess_mt(tokenizer=tokenizer, task=args.task, source=args.source, target=args.target, device=device)
    preprocess_rt = make_preprocess_rt(tokenizer=tokenizer, translations=args.mtrans, keys=args.mtrans_keys, source=args.source, device=device)

    # Load the dataset and split into training, testing and validation partitions
    logger.info(f"MACHINE TRANSLATION DATASET(S): {', '.join(args.datasetmt)}")
    if not args.local:

        if not args.opus:
            datasetmt = load_dataset(args.datasetmt[0], split=args.split) \
                .filter(lambda row: isinstance(row[args.source], str) and isinstance(row[args.target], str)) \
                .train_test_split(test_size=0.3)
        else:
            datasetmt = opus_formatter(args.datasetmt, lang1=args.source, lang2=args.target, batch_size=args.text_batch_size) \
                .train_test_split(test_size=args.test_split)
        
        # Separate the test split into test and validation partitions
        valid = datasetmt["test"].train_test_split(test_size=0.5)
        datasetmt = DatasetDict({
                    'train': datasetmt['train'],
                    'test': valid['test'],
                    'valid': valid['train'],}).shuffle(seed=42)
        
    else:
        datasetmt = load_from_disk(args.datasetmt[0])

    # Load the RT dataset
    datasetrt = load_from_disk(args.datasetrt)
    
    # Make the trainer and train the model
    if args.finetune:

        # Make separate tokenized dataset
        token_set_rt = DatasetDict({
                        'train': datasetrt['train'].map(preprocess_rt, batched=True, batch_size=args.text_batch_size),
                        'test': datasetrt['test'].map(preprocess_rt, batched=True, batch_size=args.text_batch_size),
                        'valid': datasetrt['valid'].map(preprocess_rt, batched=True, batch_size=args.text_batch_size),})

        token_set_mt = DatasetDict({
                        'train': datasetmt['train'].map(preprocess_mt, batched=True, batch_size=args.text_batch_size),
                        'test': datasetmt['test'].map(preprocess_mt, batched=True, batch_size=args.text_batch_size),
                        'valid': datasetmt['valid'].map(preprocess_mt, batched=True, batch_size=args.text_batch_size),})
        
        if args.task_mixture < 2:

            logger.info("Model finetuning started...")
            logger.info("HYPERPARAMETERS (MACHINE TRANSLATION):")
            logger.info(f"\tEPOCHS: {args.epochs / 2}")
            logger.info(f"\tBATCH SIZE: {args.batch_size}")
            logger.info(f"\tLEARNING RATE: {args.learning_rate:g}")
            logger.info(f"\tTRAINING SIZE: {len(datasetmt['train'])}")
            logger.info(f"\tVALIDATION SIZE: {len(datasetmt['valid'])}")
            logger.info(f"\tTESTING SIZE: {len(datasetmt['test'])}")
            logger.info(f"\tEVALUATION METRIC: {args.metric} using {','.join(args.metric_keys)}")

        logger.info("HYPERPARAMETERS (REPEAT TRANSLATION):")
        logger.info(f"\tEPOCHS: {args.epochs / 2 if args.task_mixture < 2 else args.epochs}")
        logger.info(f"\tBATCH SIZE: {args.batch_size}")
        logger.info(f"\tLEARNING RATE: {args.learning_rate:g}")
        logger.info(f"\tTRAINING SIZE: {len(datasetrt['train'])}")
        logger.info(f"\tVALIDATION SIZE: {len(datasetrt['valid'])}")
        logger.info(f"\tTESTING SIZE: {len(datasetrt['test'])}")
        logger.info(f"\tEVALUATION METRIC: {args.metric} using {','.join(args.metric_keys)}")

        # This is bad code, but I am unsure how the pointers get updated after each training set, so I make sure to give the make functions
        # the most up-to-date model by calling them explicitly after all previous training in that mixture is done. If they always track the
        # latest model, this code can be reduced significantly. I also put the try-catch at the end because I had some weird issues I was
        # unsure how to solve at the time, and this is expensive to retrain for just the end results.
        if args.task_mixture == 0:

            trainer_rt = make_trainer(model, tokenizer, dataset=token_set_rt, 
                                      learning_rate=args.learning_rate, epochs=args.epochs / 2, batch_size=args.batch_size, 
                                      save_at=args.save_at, output=args.output, metric_name=args.metric, metric_keys=args.metric_keys)

            if not args.skip:
                logger.info(f"PRE-EVALUATION ON REPEAT TRANSLATION (VALIDATION): {str(trainer_rt.evaluate())}")

            logger.info("FINE-TUNING ON REPEAT TRANSLATION")
            trainer_rt.train()

            trainer_mt = make_trainer(model, tokenizer, dataset=token_set_mt, 
                                      learning_rate=args.learning_rate, epochs=args.epochs / 2, batch_size=args.batch_size, 
                                      save_at=args.save_at, output=args.output, metric_name=args.metric, metric_keys=args.metric_keys)
            
            if not args.skip:
                logger.info(f"PRE-EVALUATION ON MACHINE TRANSLATION (VALIDATION): {str(trainer_mt.evaluate())}")
            
            logger.info("FINE-TUNING ON MACHINE TRANSLATION")
            trainer_mt.train()

            logger.info(f"POST-EVALUATION ON REPEAT TRANSLATION (VALIDATION): {str(trainer_rt.evaluate())}")
            logger.info(f"POST-EVALUATION ON MACHINE TRANSLATION (VALIDATION): {str(trainer_mt.evaluate())}")

        elif args.task_mixture == 1:

            trainer_mt = make_trainer(model, tokenizer, dataset=token_set_mt, 
                                      learning_rate=args.learning_rate, epochs=args.epochs / 2, batch_size=args.batch_size, 
                                      save_at=args.save_at, output=args.output, metric_name=args.metric, metric_keys=args.metric_keys)

            if not args.skip:
                logger.info(f"PRE-EVALUATION ON MACHINE TRANSLATION (VALIDATION): {str(trainer_mt.evaluate())}")

            logger.info("FINE-TUNING ON MACHINE TRANSLATION")
            trainer_mt.train()
            
            trainer_rt = make_trainer(model, tokenizer, dataset=token_set_rt, 
                                      learning_rate=args.learning_rate, epochs=args.epochs / 2, batch_size=args.batch_size, 
                                      save_at=args.save_at, output=args.output, metric_name=args.metric, metric_keys=args.metric_keys)
            if not args.skip:
                logger.info(f"PRE-EVALUATION ON REPEAT TRANSLATION (VALIDATION): {str(trainer_rt.evaluate())}")

            logger.info("FINE-TUNING ON REPEAT TRANSLATION")
            trainer_rt.train()

            logger.info(f"POST-EVALUATION ON MACHINE TRANSLATION (VALIDATION): {str(trainer_mt.evaluate())}")
            logger.info(f"POST-EVALUATION ON REPEAT TRANSLATION (VALIDATION): {str(trainer_rt.evaluate())}")

        elif args.task_mixture == 2:

            trainer_rt = make_trainer(model, tokenizer, dataset=token_set_rt, 
                                                learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batch_size, 
                                                save_at=args.save_at, output=args.output, metric_name=args.metric, metric_keys=args.metric_keys)

            if not args.skip:
                logger.info(f"PRE-EVALUATION ON REPEAT TRANSLATION (VALIDATION): {str(trainer_rt.evaluate())}")

            logger.info("FINE-TUNING ON REPEAT TRANSLATION")
            trainer_rt.train()

            trainer_mt = make_trainer(model, tokenizer, dataset=token_set_mt, 
                                      learning_rate=args.learning_rate, epochs=args.epochs / 2, batch_size=args.batch_size, 
                                      save_at=args.save_at, output=args.output, metric_name=args.metric, metric_keys=args.metric_keys)

        try:
            logger.info(f"POST-EVALUATION MACHINE TRANSLATION (TEST): {str(trainer_mt.evaluate(eval_dataset=token_set_mt['test']))}")
            logger.info(f"POST-EVALUATION REPEAT TRANSLATION (TEST): {str(trainer_rt.evaluate(eval_dataset=token_set_rt['test']))}")
        except:
            logger.info("UNABLE TO RUN POST-EVALUATION ON THE TEST SET, SKIPPING...")

        logger.info(f"SAVE LOCATION: {args.save_at}")

    # Sample the model's output
    if args.examples:

        logger.info("MACHINE TRANSLATION SAMPLES:")
        for i, row, trans in preview_translation(model, tokenizer, task=args.task, 
                                                 source=args.source, dataset=datasetmt["valid"], 
                                                 num_examples=args.examples, device=device):

            logger.info(f"TRANSLATION {i} on \"{args.task}\"")
            logger.info(f"\tOriginal Text: {row[args.source]}")
            logger.info(f"\tGenerated Translation: {trans[0]['translation_text']}")
            logger.info(f"\tActual Translation: {row[args.target]}")

        logger.info("REPEAT TRANSLATION SAMPLES:")
        for i, row, trans in preview_translation(model, tokenizer, task="Repeat the best translation", 
                                                 source=args.source, dataset=datasetrt["valid"], 
                                                 num_examples=args.examples, device=device):

            logger.info(f"REPEAT TRANSLATION {i}")
            logger.info(f"\tOriginal Text: {row[args.source]}")
            logger.info(f"\tGenerated Translation: {trans[0]['translation_text']}")

    logger.info("RUN COMPLETED")

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
        "-dmt",
        "--datasetmt",
        type=str,
        nargs="+",
        default="hackathon-pln-es/Axolotl-Spanish-Nahuatl",
        help="Datasets for the MT task; either on the HuggingFace Hub or a local directory. Assumes one dataset if opus is False.\n \n",
    )

    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="",
        help="Split to load; empty string for the full dataset.\n \n",
    )

    parser.add_argument(
        "-lc",
        "--local",
        type=int,
        default=0,
        help="If it is a local dataset; defaults to False (0).\n \n",
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
        "-drt",
        "--datasetrt",
        type=str,
        default="hackathon-pln-es/Axolotl-Spanish-Nahuatl",
        help="Datasets for the RT task; must be a local directory. Assumes one dataset if opus is False.\n \n",
    )

    parser.add_argument(
        "-mts",
        "--mtrans",
        type=str,
        nargs="+",
        help="Column names of the translations for the RT task.\n \n",
    )

    parser.add_argument(
        "-mtk",
        "--mtrans_keys",
        type=str,
        nargs="+",
        help="Column name of the translation scores for the RT task.\n \n",
    )

    parser.add_argument(
        "-tb",
        "--text_batch_size",
        type=int,
        default=32,
        help="batch size for text preprocessing.\n \n",
    )

    parser.add_argument(
        "-ts",
        "--test_split",
        type=float,
        default=0.3,
        help="Test/dev split; defaults to 0.3, 30%%.\n \n",
    )

    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="Spanish to Nahuatl",
        help="Initial prompt for the task name.\n \n",
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
        nargs="+",
        default="score",
        help="Specific keys in the metric dictionary to evaluate the model on.\n \n",
    )

    parser.add_argument(
        "-sk",
        "--skip",
        type=int,
        default=0,
        help="Skip pre-evaluation (defaults to 0).\n \n",
    )

    parser.add_argument(
        "-f",
        "--finetune",
        type=int,
        default=1,
        help="Perform model fine-tuning.\n \n",
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=4e-5,
        help="Learning rate; defaults to 4E-5.\n \n",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=2,
        help="Training epochs; defaults to 1 (greater than this should be even as to split epochs between the two tasks).\n \n",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size during preprocessing and fine-tuning.\n \n",
    )

    parser.add_argument(
        "-tm",
        "--task_mixture",
        type=int,
        default=0,
        help="How to train on the two tasks: 0 = RT-MT, 1 = MT-RT.\n \n",
    )

    parser.add_argument(
        "-sa",
        "--save_at",
        type=float,
        default=0.5,
        help="Save every X percent; for example: 0.3 saves at 0.3, 0.6 and 0.9. Also evaluates.\n \n",
    )

    parser.add_argument(
        "-x",
        "--examples",
        type=int,
        default=7,
        help="Output examples from the testing data, after fine-tuning if enabled.\n \n",
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

    # Fix error I guess... Idk why
    if 'PYTHONHOME' in os.environ:
        del os.environ['PYTHONHOME']

    parser = argparse.ArgumentParser(
        prog="finetune_repeat.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Fine-tunes a model with our dual-task-based approach.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    # Clear cuda cache
    torch.cuda.empty_cache()

    main(args)
