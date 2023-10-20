

## Taylor Swift Lyric Generation using fine-tuned LLM (distil-GPT-2 for now)

#### Task: Lyrics Generation from Taylor Swift Albums (from Kaggle)

- The original task can be seen in the Kaggle website, along with the dataset. [See here.](https://www.kaggle.com/datasets/ishikajohari/taylor-swift-all-lyrics-30-albums)

- We will take a *Text Generation* approach to the problem; using *causal language modelling* LLMs. Note that a *Seq2Seq* approach could have been used.

- We will fine-tune a pre-existing model like GPT-2 or distil-GPT-2 using a NVIDIA T4 GPU.


#### Jupyter Notebooks (or Google Colab)

- `lyrics_taylor_swift_preprocess_data_and_finetune_LLM.ipynb` used to run data-preprocessing.

- `lyrics_tayor_swift_generate_from_saved_LLM.ipynb` used to fine-tune pre-trained LLM model (distil-GPT-2).

- `Exploratory_Analysis_Taylor_Swift_Albums_Lyrics.ipynb` initial exploration on the dataset, and contains early finetuning of a model just for experimentation purposes.

Note: it is fine to focus on the first two notebooks, the last only included in case the reader needs to explore
the dataset in more detail.


#### Python Files

The scripts should be used by first applying pre-processing (`data_process.py`) ; then dataset tokenization for use in causal language modelling (`data_tokenization.py`); then training or fine-tuning the pre-trained LLM on the taylor swift dataset.

- `data_process.py` contains the data pre-processing needed and saves to a .json file

- `data_tokenization.py` defines the tokenizer and huggingface datasets (train and eval) to be used

- `train_model.py` defines the code (using huggingface transformers package) used to fine-tune the LLM on the datasets defined above.
