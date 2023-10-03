# Literally just to see what some of these methods output
import evaluate

if __name__ == "__main__":

    metric = evaluate.load_metric("blue")

    print(metric.compute([["Here we are"], ["This is more text"]], [["Here you are"], ["This is more text"]]))

