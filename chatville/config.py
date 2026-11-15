"""Configurações do Chat Ville.

Cada valor é lido nesta ordem:
1. variável de ambiente (no modo local, vem do arquivo .env);
2. st.secrets (no Streamlit Community Cloud, vem do painel "Secrets");
3. valor padrão definido aqui.
"""
import os
from pathlib import Path

try:  # python-dotenv só é instalado no modo local (requirements-local.txt)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

RAIZ = Path(__file__).resolve().parent.parent


def ler(nome: str, padrao: str | None = None) -> str | None:
    valor = os.getenv(nome)
    if valor:
        return valor
    try:
        import streamlit as st
        if nome in st.secrets:
            return str(st.secrets[nome])
    except Exception:  # sem secrets.toml ou fora do Streamlit
        pass
    return padrao


# ---------------------------------------------------------------------------
# Modo de execução
# ---------------------------------------------------------------------------
MODO = (ler("MODO", "local") or "local").strip().lower()
if MODO not in ("local", "cloud"):
    raise ValueError(f"MODO deve ser 'local' ou 'cloud', recebido: {MODO!r}")

# ---------------------------------------------------------------------------
# Banco vetorial e embeddings
# ---------------------------------------------------------------------------
# O embedding pertence ao BANCO, não ao LLM: trocar de provedor de chat não muda
# o embedding. Se mudar EMBEDDING_MODEL, recrie o banco com `python ingestao.py`.
EMBEDDING_MODEL = ler("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
COLLECTION_NAME = ler("COLLECTION_NAME", "chatville")
RETRIEVER_K = int(ler("RETRIEVER_K", "3"))
CHROMA_DIR = ler("CHROMA_DIR", str(RAIZ / "chroma_db"))
SOURCE_TEXT = ler("SOURCE_TEXT", str(RAIZ / "dados" / "joinville.md"))


def qdrant_credenciais() -> tuple[str | None, str | None]:
    """(url, api_key) do Qdrant Cloud. Lido sob demanda para a chave não ficar exposta no módulo."""
    return ler("QDRANT_URL"), ler("QDRANT_API_KEY")


# ---------------------------------------------------------------------------
# Servidor MCP (https://github.com/Romanatt0/MCP-SERVER)
# ---------------------------------------------------------------------------
# No modo cloud não há padrão: o Streamlit Cloud não enxerga o localhost da sua
# máquina, então é preciso publicar o servidor e informar a URL pública.
MCP_URL = ler("MCP_URL", "http://localhost:8000/mcp" if MODO == "local" else None)

# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------
HISTORY_LIMIT = int(ler("HISTORY_LIMIT", "10"))
MAX_TOKENS = int(ler("MAX_TOKENS", "1024"))   # evita respostas em loop infinito
TIMEOUT_SEGUNDOS = int(ler("TIMEOUT_SEGUNDOS", "60"))

SYSTEM_PROMPT = (
    "Você é o Bot-Ville, um assistente amigável que responde somente perguntas "
    "sobre a cidade de Joinville, Santa Catarina. Responda sempre em português.\n\n"
    "Regras:\n"
    "1. Para qualquer pergunta sobre Joinville (história, geografia, economia, "
    "cultura, turismo, dados da cidade), chame a ferramenta buscar_info_joinville "
    "antes de responder. Use palavras-chave na consulta, por exemplo "
    "'data de fundação', 'população', 'Festival de Dança'.\n"
    "2. Responda apenas com o que as ferramentas retornarem. Se a informação não "
    "estiver lá, diga que não encontrou — nunca invente.\n"
    "3. Para clima, temperatura ou chuva, use a ferramenta de previsão do tempo, "
    "se ela estiver disponível.\n"
    "4. Se a pergunta não for sobre Joinville, recuse educadamente.\n"
    "5. Use o histórico da conversa para entender perguntas de continuação."
)

# ---------------------------------------------------------------------------
# Provedores de LLM (lista editável em modelos.toml)
# ---------------------------------------------------------------------------
MODELOS_TOML = RAIZ / "modelos.toml"


def provedores_disponiveis() -> dict[str, dict]:
    """{id: {"nome", "modelos", ...}} na ordem do arquivo, sem os exclusivos do modo local no cloud."""
    with open(MODELOS_TOML, "rb") as f:
        provedores = tomllib.load(f)
    return {
        pid: dados for pid, dados in provedores.items()
        if not (MODO == "cloud" and dados.get("somente_local"))
    }
