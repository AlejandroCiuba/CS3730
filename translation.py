# pip install -q transformers accelerate
# Use a pipeline as a high-level helper
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM,)
                          # pipeline,)

import torch

text = "Translate from English to French: I love the way you dance and sing! It makes my heart happy."
task = ""

tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)

input = tokenizer(text=text, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
dec = tokenizer(text_target=task, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
dec = model._shift_right(dec.input_ids)
# print(input)

output = model(input_ids=input.input_ids, decoder_input_ids=dec)
clean = output.logits.squeeze()
print(clean.size())

softmax = torch.softmax(clean, dim=1)
ind = torch.max(clean, dim=1).indices

translated = tokenizer.decode(token_ids=output[0], skip_special_tokens=True)
print(translated)
