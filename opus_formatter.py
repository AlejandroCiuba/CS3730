# Convert Opus datasets into a format suitable for our fine-tuning pipeline
# Alejandro Ciuba, alc307@pitt.edu
import datasets

def opus_formatter(dataset_list, lang1 = "en", lang2 = "es") -> datasets.Dataset:

    def preprocess(lines, **kwargs):

        lines[lang1] = [translation[lang1] for translation in lines["translation"]]
        lines[lang2] = [translation[lang2] for translation in lines["translation"]]
        lines["id"]  = [f"{kwargs['name']}-{id}" for id in lines["id"]]

        return lines

    
    dsets = [datasets.load_dataset(name, lang1=lang1, lang2=lang2, split="train").map(preprocess, fn_kwargs={"name": name}, batch_size=32, batched=True).remove_columns("translation") for name in dataset_list]
    return datasets.combine.concatenate_datasets(dsets=dsets)

if __name__ == "__main__":
    print(opus_formatter({"opus_books", "opus_wikipedia"}))
