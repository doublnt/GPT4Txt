from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import LlamaCppEmbeddings
from sys import argv


def main():
    loader = TextLoader(argv[1], encoding="utf8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    llama = LlamaCppEmbeddings(model_path="./models/ggml-model-q4_0.bin")

    qdrant = Qdrant.from_documents(texts,
                                   llama,
                                   path="./db",
                                   collection_name="test",)


if __name__ == "__main__":
    main()
