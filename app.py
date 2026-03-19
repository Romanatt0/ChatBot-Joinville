import asyncio
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from pathlib import Path
import base64
import nest_asyncio
nest_asyncio.apply() 

st.set_page_config(page_title="Chat Ville", layout="wide")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def build_prompt(context: str) -> SystemMessage:
    content = (
        "Você é um assistente amigável chamado Bot-Ville que responderá somente "
        "sobre perguntas relacionadas à cidade de Joinville/Santa Catarina. "
        "Seu objetivo é fornecer informações úteis e precisas sobre a cidade. "
        f"Com base neste contexto:\n{context}\n\n"
        "Responda de forma detalhada somente com os dados fornecidos a você."
    )
    return SystemMessage(content=content)

async def get_response(agent, retriever, user_text: str) -> str:
    try:
        docs = retriever.invoke(user_text)
        context = "\n".join([d.page_content for d in docs])

        agent_with_context = create_react_agent(
            agent.nodes["agent"].bound,  
            agent.nodes["tools"].tools_by_name.values(),
            prompt=build_prompt(context),
        )

        response = await agent_with_context.ainvoke(
            {"messages": [{"role": "user", "content": user_text}]}
        )
        return response["messages"][-1].content

    except Exception as e:
        return f"Erro: {e}"

async def main():

    with open('.streamlit/custom.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    img_path = "images/joinville-img.jpg"
    if Path(img_path).exists():
        img_base64 = get_base64(img_path)
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

    st.markdown(
        """
        <style>


        .stAppViewContainer.appview-container.st-emotion-cache-1yiq2ps.e1td4qo60 {
            border-radius: 10px; /* Arredonda todos os 4 cantos com 10px */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Chat Ville — Guia de Joinville")

    
    with st.sidebar:
        st.header("Configuração")
        model = st.selectbox(
            "Modelo",
            ["qwen2.5:3b", "gemma:7b", "cnmoro/mistral_7b_portuguese:q5_K_M"],
            index=0
        )

    
    @st.cache_resource
    def load_resources():
        embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="chatville"
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever = load_resources()

    client = MultiServerMCPClient({
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        }
    })
    tools = await client.get_tools()
    llm = ChatOllama(model=model, temperature=0)
    agent = create_react_agent(llm, tools)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Conversa")
        for msg in st.session_state.messages:
            prefix = "**Você:**" if msg["role"] == "user" else "**Bot:**"
            st.markdown(f"{prefix} {msg['content']}")

        user_input = st.text_input("Digite sua mensagem")

    if st.button("Enviar") and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Gerando resposta..."):
            docs = retriever.invoke(user_input)
            context = "\n".join([d.page_content for d in docs])

            agent_ctx = create_react_agent(
                llm, tools, prompt=build_prompt(context)
            )

            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(
                agent_ctx.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]}
                )
            )
            reply = response["messages"][-1].content

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

        if st.button("Limpar histórico"):
            st.session_state.messages = []
            st.rerun()

    with col2:
        st.subheader("Ações")
        st.write("Faça perguntas ao bot sobre Joinville.")

if __name__ == "__main__":
    asyncio.run(main())