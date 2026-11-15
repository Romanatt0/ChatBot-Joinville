"""Banco vetorial e embeddings.

- Modo local: Chroma (pasta chroma_db/).
- Modo cloud: Qdrant Cloud.

Os embeddings são os mesmos nos dois modos (FastEmbed, roda na CPU, sem API) e
pertencem ao banco: trocar o provedor do LLM não muda nada aqui. Cada trecho
gravado guarda o nome do modelo de embedding em `metadata["embedding_model"]`,
para o app detectar um banco criado com outro modelo.
"""
from langchain_core.documents import Document

from chatville import config

DESTINOS = ("chroma", "qdrant")

# Distância de cosseno nos dois bancos. O padrão do Chroma é L2, que com vetores
# não normalizados (caso do FastEmbed) dá peso ao "tamanho" do vetor e piora a busca.
_CHROMA_METADATA = {"hnsw:space": "cosine"}


def destino_do_modo() -> str:
    return "chroma" if config.MODO == "local" else "qdrant"


def get_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)


def _qdrant_conexao() -> dict:
    url, api_key = config.qdrant_credenciais()
    if not url:
        raise RuntimeError("QDRANT_URL não configurada (.env ou secrets).")
    return {"url": url, "api_key": api_key}


# ---------------------------------------------------------------------------
# Leitura (usada pelo app)
# ---------------------------------------------------------------------------

def abrir_banco(embeddings, destino: str | None = None, qdrant_client=None):
    """Abre o banco existente só para consulta."""
    destino = destino or destino_do_modo()
    if destino == "chroma":
        from langchain_chroma import Chroma
        return Chroma(persist_directory=config.CHROMA_DIR, embedding_function=embeddings,
                      collection_name=config.COLLECTION_NAME)
    if destino == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        if qdrant_client is not None:
            return QdrantVectorStore(client=qdrant_client, collection_name=config.COLLECTION_NAME,
                                     embedding=embeddings)
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings, collection_name=config.COLLECTION_NAME, **_qdrant_conexao())
    raise ValueError(f"Destino desconhecido: {destino!r}")


def verificar_banco(vectorstore) -> str | None:
    """Devolve um aviso se o banco estiver vazio ou criado com outro embedding."""
    recriar = "Rode `python ingestao.py` para (re)criar o banco."
    try:
        docs = vectorstore.similarity_search("Joinville", k=1)
    except Exception as e:  # ex.: dimensão do vetor diferente
        return f"Não consegui consultar o banco vetorial ({type(e).__name__}). {recriar}"
    if not docs:
        return f"O banco vetorial está vazio. {recriar}"
    usado = docs[0].metadata.get("embedding_model")
    if usado != config.EMBEDDING_MODEL:
        return (f"O banco foi criado com o embedding `{usado or 'desconhecido'}`, mas o app está "
                f"configurado com `{config.EMBEDDING_MODEL}`. {recriar}")
    return None


# ---------------------------------------------------------------------------
# Escrita (usada só pelo ingestao.py)
# ---------------------------------------------------------------------------

def gravar(docs: list[Document], embeddings, destino: str, qdrant_client=None) -> None:
    """Recria a coleção do zero com os documentos."""
    for d in docs:
        d.metadata["embedding_model"] = config.EMBEDDING_MODEL

    if destino == "chroma":
        from langchain_chroma import Chroma
        Chroma(persist_directory=config.CHROMA_DIR, embedding_function=embeddings,
               collection_name=config.COLLECTION_NAME).delete_collection()
        Chroma.from_documents(docs, embeddings, persist_directory=config.CHROMA_DIR,
                              collection_name=config.COLLECTION_NAME,
                              collection_metadata=_CHROMA_METADATA)
    elif destino == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        if qdrant_client is not None:
            from qdrant_client.models import Distance, VectorParams
            dim = len(embeddings.embed_query("teste"))
            if qdrant_client.collection_exists(config.COLLECTION_NAME):
                qdrant_client.delete_collection(config.COLLECTION_NAME)
            qdrant_client.create_collection(config.COLLECTION_NAME,
                                            vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            QdrantVectorStore(client=qdrant_client, collection_name=config.COLLECTION_NAME,
                              embedding=embeddings).add_documents(docs)
        else:
            QdrantVectorStore.from_documents(docs, embeddings, collection_name=config.COLLECTION_NAME,
                                             force_recreate=True, **_qdrant_conexao())
    else:
        raise ValueError(f"Destino desconhecido: {destino!r}")
