import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def group_texts(examples, block_size=1024):
    # NOTE: copied from HuggingFace documentation on Transformers
    # GPT-2 max is 1024 # option 128 or other "power of 2"

    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_dataset_from_json(path_to_file: str) -> pd.DataFrame:
    return pd.read_json(path_to_file, orient="split", compression="infer")


class TokenizedDataset:
    def __init__(
        self,
        path_to_dataset: str,
        test_size_ratio: float = 0.2,
        model_type: str = "distilgpt2",
        block_size: int = 1024,
    ):
        # NOTE: max block_size for GPT-2 is the context window (1024), can use other "power of 2" size (e.g. 128)

        # Load dataset from .json file with lyrics
        self.dataset_df = load_dataset_from_json(path_to_dataset)

        # Get tokenizer for this type of Transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Transform DataFrame dataset into HuggingFace (HF) dataset
        self.dataset_all_hf = Dataset.from_list(
            [{"text": cur_lyrics} for cur_lyrics in self.dataset_df.lyrics]
        )
        # Split HF dataset into train / test split
        self.dataset_all_hf = self.dataset_all_hf.train_test_split(
            test_size=test_size_ratio
        )
        # Apply tokenization to the text HF dataset (and after removes the "text" feature)
        tokenization = lambda elem: self.tokenizer(elem["text"])
        self.dataset_all_hf = self.dataset_all_hf.map(
            tokenization, batched=True, remove_columns=["text"]
        )

        # Concatenate all the sequences and split into block_size (must be equal or smaller to model context window)
        # --> this dataset will be used for the causal language model (CLM) for learning lyrics
        f_group_texts = lambda examples: group_texts(examples, block_size=block_size)

        self.lm_dataset_all_hf = self.dataset_all_hf.map(
            f_group_texts, batched=True, num_proc=4
        )

    def get_causal_LM_hf_dataset(self):
        # gets HF causal language model dataset
        return self.lm_dataset_all_hf

    def get_original_hf_dataset(self):
        # gets HF original dataset: before block_size application.
        return self.dataset_all_hf

    def get_tokenizer(self):
        # gets HF model used
        return self.tokenizer

    def get_dataframe_dataset(self):
        # gets loaded dataset from path_to_file
        return self.dataset_df
