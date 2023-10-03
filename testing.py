# Literally just to see what some of these methods output
import evaluate

if __name__ == "__main__":

    metric = evaluate.load("bleu")

    print(results := metric.compute(predictions=["Here you are", "This is more text"], references=[["Here we are"], ["This is more text"]]))

    print({k: round(v, 4) for k, v in results.items() if isinstance(v, (float, int))})

