import asyncio
import base64
import threading
from pathlib import Path

import streamlit as st
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama, OllamaEmbeddings

import config

st.set_page_config(page_title="Chat Ville", layout="wide")


# ---------------------------------------------------------------------------
# Recursos (criados uma vez e reaproveitados entre os reruns do Streamlit)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _event_loop():
    # Um único loop em segundo plano: os clientes HTTP assíncronos (Ollama, MCP)
    # ficam presos ao loop em que foram criados, então não dá para usar
    # asyncio.run() (que cria e fecha um loop novo) a cada chamada.
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _event_loop()).result()


@st.cache_resource(show_spinner=False)
def load_retriever():
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})


@st.cache_resource(show_spinner=False)
def _load_mcp_tools():
    # Exceções não são cacheadas: se o servidor estiver fora, tenta de novo no próximo rerun.
    client = MultiServerMCPClient({
        "joinville": {"transport": "http", "url": config.MCP_URL},
    })
    return run_async(client.get_tools())


def get_mcp_tools():
    """Retorna (ferramentas, erro). Sem servidor MCP o app continua só com o RAG."""
    try:
        return _load_mcp_tools(), None
    except Exception as e:  # inclui ExceptionGroup do anyio
        return [], str(e)


def make_rag_tool(retriever):
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


@st.cache_resource(show_spinner=False)
def load_agent(model: str, mcp_tool_names: tuple, _mcp_tools: list, _retriever):
    """Retorna (agente, ferramenta de busca)."""
    # mcp_tool_names entra na chave do cache; parâmetros com "_" são ignorados pelo Streamlit.
    # num_predict limita o tamanho da resposta: sem ele, modelos pequenos às vezes
    # entram em loop repetindo texto e a resposta nunca termina.
    llm = ChatOllama(model=model, temperature=0, num_predict=1024)
    rag_tool = make_rag_tool(_retriever)
    agent = create_agent(llm, [rag_tool, *_mcp_tools], system_prompt=config.SYSTEM_PROMPT)
    return agent, rag_tool


# ---------------------------------------------------------------------------
# Lógica do chat
# ---------------------------------------------------------------------------

def _tools_called(messages) -> list[str]:
    return [call["name"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]


_REFUSAL_MARKERS = ("recus", "não posso ajudar", "não posso responder", "só posso",
                    "somente sobre joinville", "apenas sobre joinville", "não é sobre joinville")


def _is_refusal(text: str) -> bool:
    return any(marker in text.lower() for marker in _REFUSAL_MARKERS)


def answer(agent, rag_tool, history: list[dict]) -> tuple[str, list[str]]:
    """Envia as últimas mensagens ao agente. Retorna (resposta, ferramentas usadas)."""
    valid = [m for m in history if not m.get("error")]
    messages = [{"role": m["role"], "content": m["content"]}
                for m in valid[-config.HISTORY_LIMIT:]]
    response = run_async(agent.ainvoke({"messages": messages}))
    tools_used = _tools_called(response["messages"][len(messages):])

    if not tools_used and not _is_refusal(response["messages"][-1].text):
        # Busca de reserva: modelos pequenos às vezes respondem "de cabeça" sem
        # consultar nada (e o Ollama não suporta tool_choice para forçar). Nesse caso
        # fazemos a busca pelo agente, com a própria pergunta, e ele responde de novo.
        # Se o modelo recusou (pergunta fora do tema), mantemos a recusa.
        question = messages[-1]["content"]
        call = {"name": rag_tool.name, "args": {"consulta": question}, "id": "busca-reserva"}
        result = run_async(rag_tool.ainvoke(call["args"]))
        messages += [
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content=result, tool_call_id=call["id"]),
        ]
        response = run_async(agent.ainvoke({"messages": messages}))
        tools_used = [f"{rag_tool.name} (automática)"]
        tools_used += _tools_called(response["messages"][len(messages):])

    return response["messages"][-1].text, tools_used


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def apply_styles():
    css_path = Path(".streamlit/custom.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

    img_path = Path("images/joinville-img.jpg")
    if img_path.exists():
        img_base64 = base64.b64encode(img_path.read_bytes()).decode()
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpg;base64,{img_base64});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background-color: rgba(0,0,0,0.5);
                z-index: 0;
            }}
            .main {{ position: relative; z-index: 1; }}
            </style>
        """, unsafe_allow_html=True)


def main():
    apply_styles()
    st.title("Chat Ville — Guia de Joinville")

    with st.sidebar:
        st.header("Configuração")
        model = st.selectbox("Modelo", config.CHAT_MODELS, index=0)

    retriever = load_retriever()
    tools, mcp_error = get_mcp_tools()
    agent, rag_tool = load_agent(model, tuple(t.name for t in tools), tools, retriever)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Conversa")
        for msg in st.session_state.messages:
            prefix = "**Você:**" if msg["role"] == "user" else "**Bot:**"
            st.markdown(f"{prefix} {msg['content']}")
            if msg.get("tools"):
                st.caption("🔧 Ferramentas usadas: " + ", ".join(msg["tools"]))

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Digite sua mensagem")
            submit = st.form_submit_button("Enviar")

        if submit and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Gerando resposta..."):
                try:
                    reply, tools_used = answer(agent, rag_tool, st.session_state.messages)
                    bot_msg = {"role": "assistant", "content": reply, "tools": tools_used}
                except Exception as e:
                    bot_msg = {"role": "assistant", "content": f"⚠️ Erro ao gerar resposta: {e}",
                               "error": True}
            st.session_state.messages.append(bot_msg)
            st.rerun()

    with col2:
        st.subheader("Ações")
        st.write("Faça perguntas ao bot sobre Joinville.")

        if st.button("Limpar histórico"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        if mcp_error:
            st.warning("Servidor MCP indisponível — respondendo só com a base local.")
            if st.button("Tentar reconectar"):
                st.rerun()
        else:
            st.success(f"MCP conectado ({len(tools)} ferramenta(s))")


if __name__ == "__main__":
    main()
