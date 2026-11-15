import asyncio
import base64
import threading
from pathlib import Path

import streamlit as st
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama, OllamaEmbeddings

import config

st.set_page_config(page_title="Chat Ville", layout="wide")


# ---------------------------------------------------------------------------
# Recursos (criados uma vez e reaproveitados entre os reruns do Streamlit)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _event_loop():
    # Um único loop persistente: o agente e as ferramentas MCP guardam clientes
    # assíncronos presos ao loop em que foram criados. Com asyncio.run() a cada
    # pergunta, o loop anterior é fechado e a 2ª mensagem falha com
    # "Event loop is closed".
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


@st.cache_resource(show_spinner=False)
def load_agent(model: str, tool_names: tuple, _tools: list):
    # tool_names entra na chave do cache; _tools (não-hasheável) é ignorado pelo Streamlit.
    llm = ChatOllama(model=model, temperature=0)
    return create_agent(llm, _tools, system_prompt=config.SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Lógica do chat
# ---------------------------------------------------------------------------

def build_user_message(context: str, question: str) -> str:
    return (
        f"Contexto sobre Joinville:\n{context}\n\n"
        f"Pergunta: {question}"
    )


def answer(agent, retriever, question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    response = run_async(agent.ainvoke(
        {"messages": [{"role": "user", "content": build_user_message(context, question)}]}
    ))
    return response["messages"][-1].text


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
    agent = load_agent(model, tuple(t.name for t in tools), tools)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Conversa")
        for msg in st.session_state.messages:
            prefix = "**Você:**" if msg["role"] == "user" else "**Bot:**"
            st.markdown(f"{prefix} {msg['content']}")

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Digite sua mensagem")
            submit = st.form_submit_button("Enviar")

        if submit and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Gerando resposta..."):
                try:
                    reply = answer(agent, retriever, user_input)
                except Exception as e:
                    reply = f"⚠️ Erro ao gerar resposta: {e}"
            st.session_state.messages.append({"role": "assistant", "content": reply})
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
