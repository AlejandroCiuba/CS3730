# pip install -q transformers accelerate
from datasets import load_dataset
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM,
                          pipeline,)
from transformers.pipelines.pt_utils import KeyPairDataset

import evaluate
import nltk
import torch

def compute_metrics(preds, labels):
    results = metric.compute(predictions=[preds], references=[[labels]])
    return {k: round(v, 4) for k, v in results.items() if isinstance(v, (float, int))}

def task_setup(example):

    # Create the task
    command = "Translate from English to French: "
    task = command + example["original_version"].replace("\n", " ")
    task = task[:sum(len(word) for word in nltk.word_tokenize(task)[:256])]  # Get the first 256 words with no format changes
    example["task"] = task

    # Reformat the target language
    lang = example["french_version"].replace("\n", " ")[:sum(len(word) for word in nltk.word_tokenize(example["french_version"], language="french"))]
    example["french_version"] = lang

    return example


dataset = load_dataset("Nicolas-BZRD/Original_Songs_Lyrics_with_French_Translation", split="train")

dataset = dataset.filter(lambda x: x['language'] == "en").map(task_setup, batch_size=32).train_test_split(test_size=0.3)

metric = evaluate.load("bleu")

tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False, model_max_length=256)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer, use_fast=False)
print(dataset)

for row in dataset["test"]:
    
    trans = pipe(row["task"])
    # print(compute_metrics(trans, row["french_version"]))
    print(row)
    print(trans)
    break


# input = tokenizer(text=text, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
# dec = tokenizer(text_target=task, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
# dec = model._shift_right(dec.input_ids)
# # print(input)

# output = model(input_ids=input.input_ids, decoder_input_ids=dec)
# clean = output.logits.squeeze()
# print(clean.size())

# softmax = torch.softmax(clean, dim=1)
# ind = torch.max(clean, dim=1).indices

# translated = tokenizer.decode(token_ids=output[0], skip_special_tokens=True)
# print(translated)
