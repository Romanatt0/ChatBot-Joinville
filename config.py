"""Configurações centralizadas do Chat Ville.

Todos os valores podem ser sobrescritos por variáveis de ambiente.
"""
import os

# Ollama
EMBEDDING_MODEL = os.getenv("CHATVILLE_EMBEDDING_MODEL", "nomic-embed-text-v2-moe")
# Só modelos com suporte a ferramentas (tool calling) funcionam com o agente.
# Confira com: ollama show <modelo>  →  Capabilities: tools
CHAT_MODELS = [
    "qwen2.5:3b",
    "llama3.1:8b",
]

# Quantas mensagens anteriores da conversa o agente recebe (memória).
HISTORY_LIMIT = int(os.getenv("CHATVILLE_HISTORY_LIMIT", "10"))

# Base vetorial (gerada pelo rag.py)
CHROMA_DIR = os.getenv("CHATVILLE_CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHATVILLE_COLLECTION", "chatville")
RETRIEVER_K = int(os.getenv("CHATVILLE_RETRIEVER_K", "3"))
SOURCE_TEXT = os.getenv("CHATVILLE_SOURCE_TEXT", "texto.txt")

# Servidor MCP (https://github.com/Romanatt0/MCP-SERVER)
MCP_URL = os.getenv("CHATVILLE_MCP_URL", "http://localhost:8000/mcp")

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
