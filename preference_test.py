# Testing to see if I can modify mT5 to our preference-based needs
from datasets import (combine,
                      DatasetDict,
                      load_from_disk,)
from transformers import (AutoModelForSeq2SeqLM, 
                          AutoTokenizer,)

import torch

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load mT5 and its tokenizer (cannot use the fast version since we didn't originally)
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", max_length=256).to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)

    # Load the toy dataset
    dataset = load_from_disk("./datasets/opus_test")
    train = dataset["train"][:8]

    # Testing a training epoch
    padding = "max_length"
    max_length = 200

    tokens = tokenizer(text=train["en"], text_target=train["es"], max_length=max_length, padding=padding, truncation=True, return_tensors="pt").to(device)

    outputs = model(input_ids=tokens.input_ids, labels=tokens.labels)

    print(outputs.loss)
    print(outputs.logits.shape)

if __name__ == "__main__":
    main()
