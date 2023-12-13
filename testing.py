# Literally just to see what some of these methods output
# Also where I put the code to get the two LLM averages
import datasets
import evaluate

if __name__ == "__main__":

    # metric = evaluate.load("bleu")

    # print(results := metric.compute(predictions=["Here you are", "This is more text"], references=[["Here we are"], ["This is more text"]]))

    # print({k: round(v, 4) for k, v in results.items() if isinstance(v, (float, int))})

    dataset = datasets.load_from_disk("datasets/ix_datasets/opus_flan_opus_nllb")
    splits = {"train", "test", "valid"}

    for split in splits:

        avg1 = sum(dataset[split]["flan-t5-large_score"]) / len(dataset[split])
        avg2 = sum(dataset[split]["nllb-200-distilled-600M_score"]) / len(dataset[split])

        print(f"===================== {split} =====================")
        print(f"flan-t5-large_score: {avg1: 0.5f}")
        print(f"nllb-200-distilled-600M_score: {avg2: 0.5f}")
