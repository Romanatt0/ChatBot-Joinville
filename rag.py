"""Gera a base vetorial (Chroma) a partir do texto.txt.

Rode sempre que alterar o texto ou o modelo de embeddings:
    python rag.py
"""
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def main():
    with open(config.SOURCE_TEXT, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_text(text)

    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    # Recria a coleção do zero para não duplicar trechos a cada execução.
    Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    ).delete_collection()

    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_name=config.COLLECTION_NAME,
    )
    print(f"{len(chunks)} trechos indexados em {config.CHROMA_DIR} (coleção '{config.COLLECTION_NAME}').")


if __name__ == "__main__":
    main()
