import argparse
import numpy as np
import os
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from data_tokenization import TokenizedDataset
from typing import Any

# import torch


def compute_perplexity(nll_loss):
    return np.clip(np.exp(nll_loss), 10**-6, 10**9)


def train_causal_lang_model(
    path_project: str,
    folder_to_save: str,
    path_to_dataset: str = "",
    model_type: str = "distilgpt2",
    tokenized_dataset_obj: TokenizedDataset | None = None,
    block_size: int = 1024,
    test_size: float = 0.2,
    save_model: bool = False,
) -> tuple[Any, Trainer, TokenizedDataset, tuple[Any]]:
    # TODO: receive as options the training arguments for the model (as config file?)

    # get tokenized dataset for model training
    if not tokenized_dataset_obj:
        data_obj = TokenizedDataset(
            path_to_dataset,
            test_size_ratio=test_size,
            model_type=model_type,
            block_size=block_size,
        )
    else:
        data_obj = tokenized_dataset_obj

    # get training and evaluation dataset, i.e. train / test split
    lm_dataset_all_hf = data_obj.get_causal_LM_hf_dataset()

    # get model tokenizer used for the dataset
    tokenizer = data_obj.get_tokenizer()

    # define data collator to fill in end of sequence if smaller than context used
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # get pre-trained model from HuggingFace Hub
    model = AutoModelForCausalLM.from_pretrained(model_type)

    # TODO: allow for training args not to be hard-coded, as an input to this function.
    # TODO: allow for automatic hyper-parameter selection?
    # NOTE: The TrainingArguments options where optimized for a NVIDIA T4 GPU

    # define training arguments for the model
    training_args = TrainingArguments(
        output_dir=os.path.join(path_project, folder_to_save),
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=0.25,
        save_total_limit=3,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=10,  # 10 ; 0.1
        learning_rate=1e-4,  # 2e-5
        push_to_hub=False,
        per_device_train_batch_size=4,  # 8 , 16
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # 16
        num_train_epochs=35,  # 20 ; 15
        warmup_steps=150,
    )
    # removed regularization: weight_decay=0.01,

    # get trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset_all_hf["train"],
        eval_dataset=lm_dataset_all_hf["test"],
        data_collator=data_collator,
    )

    # train the model now (if GPU available then runs there)
    train_result = trainer.train()

    # get final train set metrics
    metrics_train = train_result.metrics
    # compute_perplexity(metrics_train["train_loss"])

    # get final test set metrics
    metrics_eval = trainer.evaluate()
    # compute_perplexity(metrics_eval["eval_loss"])

    if save_model:
        try:
            # folder_to_save example is: "experiments/"
            model.save_pretrained(
                os.path.join(path_project, folder_to_save + "finetuned_model")
            )
            tokenizer.save_pretrained(
                os.path.join(path_project, folder_to_save + "finetuned_model")
            )
        except:
            raise (
                "Error while saving model and tokenizer, please check paths or inputs."
            )

    return model, trainer, data_obj, (metrics_train, metrics_eval)


def main():
    parser = argparse.ArgumentParser(
        description="training-transformer-causal-language-modelling"
    )

    parser.add_argument("--path-project", type=str, help="project filepath")
    parser.add_argument("--folder-to-save", type=str, help="")
    parser.add_argument("--path-to-dataset", type=str, help="")
    parser.add_argument("--model-type", default="distilgpt2", type=str, help="")
    parser.add_argument("--block-size", default=1024, type=int, help="")
    parser.add_argument("--test-size", default=0.2, type=float, help="")
    parser.add_argument("--save-model", action="store_true", default=False, help="")

    args = parser.parse_args()

    train_causal_lang_model(
        path_project=args.path_project,
        folder_to_save=args.folder_to_save,
        path_to_dataset=args.path_to_dataset,
        model_type=args.model_type,
        tokenized_dataset_obj=None,
        block_size=args.block_size,
        test_size=args.test_size,
        save_model=args.save_model,
    )


if __name__ == "__main__":
    main()
