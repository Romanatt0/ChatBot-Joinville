"""Cria (ou recria) o banco vetorial a partir de dados/joinville.md.

Uso:
    python ingestao.py                    # destino do MODO atual (local=chroma, cloud=qdrant)
    python ingestao.py --destino chroma
    python ingestao.py --destino qdrant   # precisa de QDRANT_URL e QDRANT_API_KEY
    python ingestao.py --destino ambos

Rode de novo sempre que alterar o texto ou o EMBEDDING_MODEL.

Como o texto é dividido:
- Cada seção do Markdown (## / ###) é dividida separadamente, então um trecho
  nunca mistura assuntos de seções diferentes.
- Seções que são listas (ex.: "Dados gerais") viram um trecho por item: cada fato
  fica isolado e a busca encontra exatamente ele.
- Todo trecho começa com o caminho da seção (ex.: "[Joinville > História > ...]"),
  o que dá contexto tanto para a busca quanto para o modelo.
"""
import argparse
import re
import time

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from chatville import banco, config

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
            docs.append(Document(page_content=f"[{caminho}]\n{pedaco}", metadata={"secao": caminho}))
    return docs


def carregar_documentos() -> list[Document]:
    with open(config.SOURCE_TEXT, "r", encoding="utf-8") as f:
        return dividir(limpar(f.read()))


def main():
    parser = argparse.ArgumentParser(description="Cria o banco vetorial do Chat Ville.")
    parser.add_argument("--destino", choices=[*banco.DESTINOS, "ambos"], default=banco.destino_do_modo(),
                        help="onde gravar (padrão: chroma no modo local, qdrant no modo cloud)")
    args = parser.parse_args()
    destinos = banco.DESTINOS if args.destino == "ambos" else (args.destino,)

    docs = carregar_documentos()
    print(f"{len(docs)} trechos gerados a partir de {config.SOURCE_TEXT}")
    print(f"Embedding: {config.EMBEDDING_MODEL} (o download acontece só na primeira vez)")
    embeddings = banco.get_embeddings()

    for destino in destinos:
        inicio = time.time()
        banco.gravar([Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs],
                     embeddings, destino)
        print(f"✔ {destino}: coleção '{config.COLLECTION_NAME}' recriada em {time.time() - inicio:.0f}s")


if __name__ == "__main__":
    main()
