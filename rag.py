"""Gera a base vetorial (Chroma) a partir de dados/joinville.md.

Rode sempre que alterar o texto ou o modelo de embeddings:
    python rag.py

Como o texto é dividido:
- Cada seção do Markdown (## / ###) é dividida separadamente, então um trecho
  nunca mistura assuntos de seções diferentes.
- Seções que são listas (ex.: "Dados gerais") viram um trecho por item: cada fato
  fica isolado e a busca encontra exatamente ele.
- Todo trecho começa com o caminho da seção (ex.: "[Joinville > História > ...]"),
  o que dá contexto tanto para a busca quanto para o modelo.
"""
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config

REFERENCIAS = re.compile(r"\[(\d+|carece de fontes|nota \d+)\]")  # marcas da Wikipédia


def limpar(texto: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", REFERENCIAS.sub("", texto))


def dividir(markdown: str) -> list[Document]:
    secoes = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "titulo"), ("##", "secao"), ("###", "subsecao")],
    ).split_text(markdown)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = []
    for secao in secoes:
        caminho = " > ".join(
            ["Joinville"] + [secao.metadata[k] for k in ("secao", "subsecao") if k in secao.metadata]
        )
        linhas = [l for l in secao.page_content.splitlines() if l.strip()]
        if linhas and all(l.startswith("- ") for l in linhas):
            pedacos = [l[2:] for l in linhas]  # lista de fatos: um trecho por item
        else:
            pedacos = splitter.split_text(secao.page_content)

        for pedaco in pedacos:
            docs.append(Document(
                page_content=f"[{caminho}]\n{pedaco}",
                metadata={"secao": caminho},
            ))
    return docs


def main():
    with open(config.SOURCE_TEXT, "r", encoding="utf-8") as f:
        docs = dividir(limpar(f.read()))

    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    # Recria a coleção do zero para não duplicar trechos a cada execução.
    Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    ).delete_collection()

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_name=config.COLLECTION_NAME,
    )
    print(f"{len(docs)} trechos indexados em {config.CHROMA_DIR} (coleção '{config.COLLECTION_NAME}').")


if __name__ == "__main__":
    main()
