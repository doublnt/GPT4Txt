# Environment Setup

In order to set your environment up to run the code here, first install all requirements:

```shell
conda create --name gpt4txt-env
conda activate gpt4txt-env

or

pip install -r requirements.txt
```

Then, download the 2 models and place them in a folder called `./models`:
- LLM: default to [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin). If you prefer a different GPT4All-J compatible model, just download it and reference it in `
gpt4txt.py`.
- Embedding: default to [ggml-model-q4_0.bin](https://huggingface.co/Pi3141/alpaca-native-7B-ggml/resolve/397e872bf4c83f4c642317a5bf65ce84a105786e/ggml-model-q4_0.bin). If you prefer a different compatible Embeddings model, just download it and reference it in `embedding.py` and `gpt4txt.py`.

## Test dataset
This repo uses a sushi.txt as an example.

## Instructions for ingesting your own dataset

Get your .txt file ready.

Run the following command to ingest the data.

```shell
python embedding.py <path_to_your_txt_file>
```

It will create a `db` folder containing the local vectorstore. Will take time, depending on the size of your document.
You can ingest as many documents as you want by running `embedding`, and all will be accumulated in the local embeddings database. 
If you want to start from scratch, delete the `db` folder.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python gpt4txt.py
```

And wait for the script to require your input. 