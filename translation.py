# pip install -q transformers accelerate
from datasets import load_dataset
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM,
                          pipeline,)
from transformers.pipelines.pt_utils import KeyPairDataset
from tqdm import tqdm

import evaluate
import nltk
import torch

def compute_metrics(preds, labels):
    results = metric.compute(predictions=[preds], references=[[labels]])
    return {k: round(v, 4) for k, v in results.items() if isinstance(v, (float, int))}

def task_setup(examples):

    # Create the task
    task = [example.replace("\n", " ") for example in examples["original_version"]]
    task = [t[:sum(len(word) for word in nltk.word_tokenize(t)[:256])] for t in task]  # Get the first 256 words with no format changes
    examples["task"] = task

    # Reformat the target language
    lang = [example.replace("\n", " ")[:sum(len(word) for word in nltk.word_tokenize(example, language="french")[:256])] for example in examples["french_version"]]
    examples["french_version"] = lang

    return examples


dataset = load_dataset("Nicolas-BZRD/Original_Songs_Lyrics_with_French_Translation", split="train")

dataset = dataset.filter(lambda x: x['language'] == "en").map(task_setup, batched=True, batch_size=32).train_test_split(test_size=0.3)

metric = evaluate.load("sacrebleu")

tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False, model_max_length=256)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer, use_fast=False)
print(dataset)

total = 0
count = 0
for row in tqdm(dataset["test"]):

    trans = pipe(row["task"])
    total += compute_metrics(trans[0]["translation_text"], row["french_version"])['score']
    count += 1

    if count >= 10:
        break

print(f"Final Avg BLEU Score: {total/count:.4f}")


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
