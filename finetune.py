# Finetune model
from datasets import (DatasetDict,
                      load_dataset,)
from transformers import (AutoModelForSeq2SeqLM, 
                          AutoTokenizer, 
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          pipeline,)

import argparse
import evaluate
import os
import tensorboard
import torch

import numpy as np

VERSION = "1.0.0"

def make_preprocess(tokenizer):
    """
    Returns a preprocessing function using the given tokenizer.
    """

    def preprocess(examples):
        """
        Tokenizes the inputs and targets to feed to the model.
        """

        padding = "max_length"
        max_length = 200

        inputs = [str(i) for i in examples["sp"]]
        targets = [str(t) for t in examples["nah"]]

        # Memory reqs grow quadratically with input size, stops at max_length
        tokens = tokenizer(inputs, max_length=max_length, padding=padding, truncation=True, return_tensors="pt")

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_length, padding=padding, truncation=True, return_tensors="pt")

        tokens["labels"] = labels["input_ids"]
        return tokens

    return preprocess

def make_compute_metrics(tokenizer, metrics_name="bleu"):

    metric = evaluate.load(metrics_name)

    def compute_metrics(eval_pred):

        preds, labels = eval_pred

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        with tokenizer.as_target_tokenizer():
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        results = metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {k: round(v, 4) for k, v in results.items() if isinstance(v, (float, int))}
    
    return compute_metrics

def make_trainer(model, tokenizer, dataset,
                 learning_rate = 4e-5, batch_size = 8, epochs = 1, 
                 output = "custom", metrics_name = "bleu"):
    """
    Creates a trainer for the model.
    """

    args = Seq2SeqTrainingArguments(
            output_dir=f"models/{output}",
            evaluation_strategy="steps",
            eval_steps=100,
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="steps",
            save_steps=200,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model=metrics_name,
            report_to="tensorboard")

    compute_metrics = make_compute_metrics(tokenizer=tokenizer, metrics_name=metrics_name)

    dc = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    return Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            data_collator=dc,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,)

def preview_translation(model, tokenizer, dataset, num_examples = 5):
    
    padding = "max length"
    max_length = 200

    tokens = tokenizer(dataset[:5]["sp"], max_length=max_length, padding=padding, truncation=True, return_tensors="pt")

    with tokenizer.as_target_tokenizer():
            labels = tokenizer(dataset[:5]["nah"], max_length=max_length, padding=padding, truncation=True, return_tensors="pt")

    tokens["labels"] = labels["input_ids"]

    output = model(tokens, decoder_input_ids=model._shift_right(labels.input_ids))

    with tokenizer.as_target_tokenizer():
        return tokenizer.batch_decode(output.last_hidden_state, skip_special_tokens=True)

def main(args: argparse.ArgumentParser):

    # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    # Load the preprocessing function
    preprocess = make_preprocess(tokenizer=tokenizer)

    # Load the dataset and split into training, testing and validation partitions
    dataset = load_dataset("hackathon-pln-es/Axolotl-Spanish-Nahuatl", split="train") \
        .filter(lambda row: isinstance(row['sp'], str) and isinstance(row['nah'], str)) \
        .train_test_split(test_size=0.3) \
        .map(preprocess, batched=True, batch_size = 2)
    
    # Separate the test split into test and validation partitions
    valid = dataset["test"].train_test_split(test_size=0.5)
    dataset = DatasetDict({
                'train': dataset['train'],
                'test': dataset['test'],
                'valid': valid['train']})
    
    # Make the trainer and train the model
    trainer = make_trainer(model, tokenizer, dataset, output="test_run")
    trainer.train()

    # Sample the model's output
    examples = preview_translation(model, tokenizer, dataset["test"])


def add_args(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser if split.py is the main program.
    """

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
