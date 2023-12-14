# Literally just to see what some of these methods output
# Also where I put the code to get the two LLM averages\
# Also where I got the translations from the log files
from datasets import load_dataset
from pathlib import Path

import evaluate
import re

import pandas as pd

if __name__ == "__main__":

    # metric = evaluate.load("bleu")

    # print(results := metric.compute(predictions=["Here you are", "This is more text"], references=[["Here we are"], ["This is more text"]]))

    # print({k: round(v, 4) for k, v in results.items() if isinstance(v, (float, int))})

    # dataset = datasets.load_from_disk("datasets/ix_datasets/opus_flan_opus_nllb")
    # splits = {"train", "test", "valid"}

    # for split in splits:

    #     avg1 = sum(dataset[split]["flan-t5-large_score"]) / len(dataset[split])
    #     avg2 = sum(dataset[split]["nllb-200-distilled-600M_score"]) / len(dataset[split])

    #     print(f"===================== {split} =====================")
    #     print(f"flan-t5-large_score: {avg1: 0.5f}")
    #     print(f"nllb-200-distilled-600M_score: {avg2: 0.5f}")

    # dataset = load_dataset("opus_books", lang1="en", lang2="es", split="train")
    # print(len(dataset))

    # dataset = load_dataset("opus_wikipedia", lang1="en", lang2="es", split="train")
    # print(len(dataset))

    # gpt_df = pd.read_csv("./chatgpt_translate/comparative_samples_chatgpt_translated.csv")
    # print(gpt_df.info())

    # Randomly select five sentences for our evaluation
    # print(gpt_df.sample(5))  # 58, 57, 35, 73, 34

    # indices = [34, 35, 57, 58, 73]

    # gpt_df = gpt_df.iloc[indices, :]
    # gpt_df.rename(columns={"Spanish": "GPT-3.5"}, inplace=True)
    # print(gpt_df)

    # # Get the sentences from the log files (preference = GCL and preference-alt = LCL)
    # LOGPATH = Path("./logs")
    # LOGS = {"finetune*", "preference*", "preference-alt*", "repeat*"}

    # for log in LOGS:

    #     for file in LOGPATH.glob(log):

    #         with open(file, 'r', encoding='utf-8') as src:
    #             full = [text[38:].strip() for text in src.readlines(-1)]
            
    #         selected = []
    #         for index in indices:

    #             line_locator = f"TRANSLATION {index} on \"English to Spanish\""

    #             try:
    #                 start = full.index(line_locator)
    #             except ValueError:
    #                 print(f"{line_locator} NOT FOUND IN FILE {file}")

    #             selected.append(full[start + 2].removeprefix("Generated Translation: "))

    #         gpt_df[file] = selected

    df = pd.read_csv("./chatgpt_translate/all_models.csv")

    index = 0
    for col in df.columns:
        print(f"{col.upper()}: {df.loc[0, col]}")

    # print(gpt_df.info)
    # gpt_df.to_csv("./chatgpt_translate/all_models.csv")
