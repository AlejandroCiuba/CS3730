# Finetune model
from datasets import (DatasetDict,
                      load_dataset,)
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
import os
import tensorboard
import torch

import numpy as np

VERSION = "1.0.7"

def make_preprocess(tokenizer, task = "", source = "", target = "", device = "cuda"):
    """
    Returns a preprocessing function using the given tokenizer.
    """

    def preprocess(examples):
        """
        Tokenizes the inputs and targets to feed to the model.
        """

        padding = "max_length"
        max_length = 200

        inputs = [task + ": " + str(i) for i in examples[source]]
        targets = [task + ": " + str(t) for t in examples[target]]

        # Memory reqs grow quadratically with input size, stops at max_length
        tokens = tokenizer(text=inputs, text_target=targets, max_length=max_length, padding=padding, truncation=True, return_tensors="pt").to(device)
        return tokens

    return preprocess

def make_compute_metrics(tokenizer, metrics_name="bleu"):

    metric = evaluate.load(metrics_name)

    def compute_metrics(eval_pred):

        preds, labels = eval_pred

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        results = metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {k : v for k, v in results.items()}
    
    return compute_metrics

def make_trainer(model, tokenizer, dataset,
                 learning_rate = 4e-5, epochs = 1, batch_size = 8, 
                 save_at = 0.5, output = "models", metrics_name = "bleu"):
    """
    Creates a trainer for the model.
    """

    save_steps = int(len(dataset["train"]) * save_at)

    args = Seq2SeqTrainingArguments(
            output_dir=output,
            evaluation_strategy="steps",
            eval_steps=save_steps,
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="steps",
            save_steps=save_steps,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=epochs,
            predict_with_generate=True,
            bf16=True,
            load_best_model_at_end=True,
            metric_for_best_model=metrics_name,
            report_to="tensorboard")

    compute_metrics = make_compute_metrics(tokenizer=tokenizer, metrics_name=metrics_name)

    dc = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    return Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            data_collator=dc,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,)

def preview_translation(model, tokenizer, dataset, task = "", num_examples = 5, **kwargs):
    
    pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=kwargs["device"])

    for i, row in tqdm(enumerate(dataset)):
        
        trans = pipe(task + ": " + row["sp"])

        print(f"===================== TRANSLATION {i} =====================")
        print(f"Spanish Text: {row['sp']}\n")
        print(f"Translation: {trans[0]['translation_text']}\n")
        print(f"Actual Translatin: {row['nah']}\n")

        if i == num_examples:
            break

def main(args: argparse.ArgumentParser):

    # Sync the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and tokenizer; we use a sentencepiece-based tokenizer, so we disable fast-tokenization
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, max_length=256).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Load the preprocessing function
    preprocess = make_preprocess(tokenizer=tokenizer, task=args.task, source=args.source, target=args.target, device=device)

    # Load the dataset and split into training, testing and validation partitions
    dataset = load_dataset(args.dataset, split=args.split if args.split != "" else None) \
        .filter(lambda row: isinstance(row[args.source], str) and isinstance(row[args.target], str)) \
        .train_test_split(test_size=0.3)
    
    # Separate the test split into test and validation partitions
    valid = dataset["test"].train_test_split(test_size=0.5)
    dataset = DatasetDict({
                'train': dataset['train'],
                'test': valid['test'],
                'valid': valid['train'],})
    
    # Make the trainer and train the model
    if args.finetune:

        # Make separate tokenized dataset
        token_set = DatasetDict({
                        'train': dataset['train'].map(preprocess, batched=True, batch_size=args.batchsize),
                        'test': dataset['test'].map(preprocess, batched=True, batch_size=args.batchsize),
                        'valid': dataset['valid'].map(preprocess, batched=True, batch_size=args.batchsize),})

        print("Model finetuning started...")
        trainer = make_trainer(model, tokenizer, dataset=token_set, 
                               learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batchsize, 
                               save_at=args.save_at, output=args.output, metrics_name=args.metric)

        print("===================== PRE-EVALUATION =====================")
        print(trainer.evaluate())

        print("===================== FINE-TUNING =====================")
        trainer.train()

        print("===================== POST-EVALUATION =====================")
        print(trainer.evaluate())

    # Sample the model's output
    if args.examples:
        preview_translation(model, tokenizer, task=args.task, dataset=dataset["test"], device=device)

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
        default="hackathon-pln-es/Axolotl-Spanish-Nahuatl",
        help="Dataset; either on the HuggingFace Hub or a local directory.\n \n",
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
        default=1,
        help="Training epochs; defaults to 1.\n \n",
    )

    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=8,
        help="Batch size during preprocessing and fine-tuning.\n \n",
    )

    parser.add_argument(
        "-sa",
        "--save_at",
        type=float,
        default=0.5,
        help="Save every X percent; for example: 0.3 -> Evaluates at 0.3, 0.6 and 0.9. Also evaluates.\n \n",
    )

    parser.add_argument(
        "-x",
        "--examples",
        type=int,
        default=7,
        help="Output examples from the testing data (after fine-tuning if enabled).\n \n",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Directory to save the models.\n \n",
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
        prog="segment.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Outputs a segmented .eaf file according to APLS standards.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    # Clear cuda cache
    torch.cuda.empty_cache()

    main(args)
