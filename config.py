"""Configurações centralizadas do Chat Ville.

Todos os valores podem ser sobrescritos por variáveis de ambiente.
"""
import os

# Ollama
EMBEDDING_MODEL = os.getenv("CHATVILLE_EMBEDDING_MODEL", "nomic-embed-text-v2-moe")
CHAT_MODELS = [
    "qwen2.5:3b",
    "gemma:7b",
    "cnmoro/mistral_7b_portuguese:q5_K_M",
]

# Base vetorial (gerada pelo rag.py)
CHROMA_DIR = os.getenv("CHATVILLE_CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHATVILLE_COLLECTION", "chatville")
RETRIEVER_K = int(os.getenv("CHATVILLE_RETRIEVER_K", "3"))
SOURCE_TEXT = os.getenv("CHATVILLE_SOURCE_TEXT", "texto.txt")

# Servidor MCP (https://github.com/Romanatt0/MCP-SERVER)
MCP_URL = os.getenv("CHATVILLE_MCP_URL", "http://localhost:8000/mcp")

SYSTEM_PROMPT = (
    "Você é um assistente amigável chamado Bot-Ville que responderá somente "
    "sobre perguntas relacionadas à cidade de Joinville/Santa Catarina. "
    "Seu objetivo é fornecer informações úteis e precisas sobre a cidade. "
    "Use as ferramentas disponíveis quando fizer sentido (por exemplo, para "
    "previsão do tempo). Para as demais perguntas, responda de forma detalhada "
    "somente com os dados do contexto fornecido."
)
