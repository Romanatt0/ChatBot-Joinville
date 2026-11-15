"""Agente: ferramentas (busca na base + MCP) e a lógica de resposta.

Não depende do Streamlit nem de um provedor específico: recebe um chat model
pronto (de chatville.llm.get_llm) e um retriever (de chatville.banco).
"""
import asyncio
import threading

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from chatville import config

# ---------------------------------------------------------------------------
# Event loop permanente
# ---------------------------------------------------------------------------
# Os clientes HTTP assíncronos (LLM, MCP) ficam presos ao loop em que foram
# criados, então não dá para usar asyncio.run() (que cria e fecha um loop novo)
# a cada chamada. Usamos um único loop rodando numa thread em segundo plano.
_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def run_async(coro):
    global _loop
    with _lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True).start()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# ---------------------------------------------------------------------------
# Ferramentas
# ---------------------------------------------------------------------------

def carregar_ferramentas_mcp(url: str | None) -> tuple[list, str | None]:
    """Retorna (ferramentas, erro). Sem servidor MCP o agente segue só com a busca."""
    if not url:
        return [], "MCP_URL não configurada."
    from langchain_mcp_adapters.client import MultiServerMCPClient
    try:
        client = MultiServerMCPClient({"joinville": {"transport": "http", "url": url}})
        return run_async(client.get_tools()), None
    except Exception as e:  # inclui ExceptionGroup do anyio
        return [], type(e).__name__


def criar_ferramenta_busca(retriever):
    """Expõe a base vetorial como ferramenta: o agente decide quando e o que buscar."""

    @tool
    def buscar_info_joinville(consulta: str) -> str:
        """Busca informações sobre Joinville (história, fundação, população,
        geografia, economia, cultura, turismo, eventos) na base de conhecimento.

        Args:
            consulta: termos de busca em português, ex.: "data de fundação".
        """
        docs = retriever.invoke(consulta)
        if not docs:
            return "Nenhuma informação encontrada."
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return buscar_info_joinville


def criar_agente(llm, retriever, ferramentas_mcp: list):
    """Retorna (agente, ferramenta de busca)."""
    busca = criar_ferramenta_busca(retriever)
    agente = create_agent(llm, [busca, *ferramentas_mcp], system_prompt=config.SYSTEM_PROMPT)
    return agente, busca


# ---------------------------------------------------------------------------
# Resposta
# ---------------------------------------------------------------------------

def _ferramentas_chamadas(mensagens) -> list[str]:
    return [c["name"] for m in mensagens if isinstance(m, AIMessage) for c in m.tool_calls]


_MARCAS_DE_RECUSA = ("recus", "não posso ajudar", "não posso responder", "só posso",
                     "somente sobre joinville", "apenas sobre joinville", "não é sobre joinville")


def _eh_recusa(texto: str) -> bool:
    return any(marca in texto.lower() for marca in _MARCAS_DE_RECUSA)


def responder(agente, busca, historico: list[dict]) -> tuple[str, list[str]]:
    """Envia as últimas mensagens ao agente. Retorna (resposta, ferramentas usadas)."""
    validas = [m for m in historico if not m.get("error")]
    mensagens = [{"role": m["role"], "content": m["content"]}
                 for m in validas[-config.HISTORY_LIMIT:]]
    resposta = run_async(agente.ainvoke({"messages": mensagens}))
    usadas = _ferramentas_chamadas(resposta["messages"][len(mensagens):])

    if not usadas and not _eh_recusa(resposta["messages"][-1].text):
        # Busca de reserva: alguns modelos respondem "de cabeça" sem consultar
        # nada. Nesse caso fazemos a busca pelo agente, com a própria pergunta,
        # e ele responde de novo. Se o modelo recusou (fora do tema), mantemos.
        pergunta = mensagens[-1]["content"]
        chamada = {"name": busca.name, "args": {"consulta": pergunta}, "id": "busca-reserva"}
        resultado = run_async(busca.ainvoke(chamada["args"]))
        mensagens += [
            AIMessage(content="", tool_calls=[chamada]),
            ToolMessage(content=resultado, tool_call_id=chamada["id"]),
        ]
        resposta = run_async(agente.ainvoke({"messages": mensagens}))
        usadas = [f"{busca.name} (automática)"]
        usadas += _ferramentas_chamadas(resposta["messages"][len(mensagens):])

    return resposta["messages"][-1].text, usadas
