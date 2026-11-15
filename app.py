import base64
from pathlib import Path

import streamlit as st

from chatville import agente, banco, config, llm

st.set_page_config(page_title="Chat Ville", layout="wide")

OUTRO_MODELO = "Outro (digitar)…"


# ---------------------------------------------------------------------------
# Recursos compartilhados (criados uma vez e reaproveitados entre os reruns)
# Atenção: nada que contenha a chave do usuário é cacheado aqui, porque
# st.cache_resource é compartilhado entre todas as sessões do app.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando o modelo de embeddings (só na primeira vez)…")
def carregar_banco():
    vectorstore = banco.abrir_banco(banco.get_embeddings())
    return vectorstore, banco.verificar_banco(vectorstore)


@st.cache_resource(show_spinner=False)
def _mcp_em_cache(url: str):
    ferramentas, erro = agente.carregar_ferramentas_mcp(url)
    if erro:
        raise RuntimeError(erro)  # exceções não são cacheadas: tenta de novo no próximo rerun
    return ferramentas


def ferramentas_mcp() -> tuple[list, str | None]:
    if not config.MCP_URL:
        return [], "sem_url"
    try:
        return _mcp_em_cache(config.MCP_URL), None
    except RuntimeError as e:
        return [], str(e)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

def barra_lateral() -> tuple[str, str, str, str | None]:
    """Retorna (provedor, nome do provedor, modelo, chave digitada pelo usuário)."""
    provedores = config.provedores_disponiveis()
    with st.sidebar:
        st.header("Configuração")
        st.caption(f"Modo: **{config.MODO}**")

        provedor = st.selectbox("Provedor", list(provedores),
                                format_func=lambda p: provedores[p].get("nome", p))
        dados = provedores[provedor]
        nome = dados.get("nome", provedor)

        escolha = st.selectbox("Modelo", [*dados.get("modelos", []), OUTRO_MODELO],
                               key=f"modelo_{provedor}")
        modelo = escolha
        if escolha == OUTRO_MODELO:
            modelo = st.text_input("Nome do modelo", key=f"modelo_custom_{provedor}",
                                   placeholder="ex.: nome-exato-do-modelo").strip()
            st.caption("O modelo precisa suportar ferramentas (tool calling).")

        chave = None
        if llm.precisa_de_chave(provedor):
            # A chave fica só em st.session_state desta sessão: não é salva,
            # registrada em log nem exibida.
            chaves = st.session_state.setdefault("chaves", {})
            digitada = st.text_input(f"Chave de API do {nome}", type="password",
                                     value=chaves.get(provedor, ""), key=f"campo_chave_{provedor}",
                                     help="Fica só nesta sessão do navegador. Não é salva.")
            chaves[provedor] = digitada.strip()
            chave = chaves[provedor] or None

            if chave:
                st.caption("🔒 Usando a sua chave.")
            elif llm.chave_do_app(provedor):
                st.caption("Usando a chave do app. Informe a sua se atingir o limite de uso.")
            else:
                st.warning(f"Informe sua chave de API do {nome}.")

    return provedor, nome, modelo, chave


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def aplicar_estilos():
    css = Path(".streamlit/custom.css")
    if css.exists():
        st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

    img = Path("images/joinville-img.jpg")
    if img.exists():
        img_base64 = base64.b64encode(img.read_bytes()).decode()
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


def gerar_resposta(provedor, nome, modelo, chave, retriever, ferramentas) -> dict:
    try:
        modelo_chat = llm.get_llm(provedor, modelo, chave)
        ag, busca = agente.criar_agente(modelo_chat, retriever, ferramentas)
        texto, usadas = agente.responder(ag, busca, st.session_state.messages)
        return {"role": "assistant", "content": texto, "tools": usadas}
    except Exception as e:
        return {"role": "assistant", "content": llm.mensagem_de_erro(e, provedor, nome), "error": True}


def main():
    aplicar_estilos()
    st.title("Chat Ville — Guia de Joinville")

    provedor, nome, modelo, chave = barra_lateral()

    try:
        vectorstore, aviso_banco = carregar_banco()
    except Exception as e:
        st.error(f"Não consegui abrir o banco vetorial ({type(e).__name__}). "
                 "Confira a configuração e rode `python ingestao.py`.")
        st.stop()
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
    ferramentas, erro_mcp = ferramentas_mcp()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Conversa")
        if aviso_banco:
            st.warning(aviso_banco)
        for msg in st.session_state.messages:
            prefixo = "**Você:**" if msg["role"] == "user" else "**Bot:**"
            st.markdown(f"{prefixo} {msg['content']}")
            if msg.get("tools"):
                st.caption("🔧 Ferramentas usadas: " + ", ".join(msg["tools"]))

        with st.form(key="chat_form", clear_on_submit=True):
            pergunta = st.text_input("Digite sua mensagem")
            enviar = st.form_submit_button("Enviar")

        if enviar and pergunta:
            if not modelo:
                st.warning("Digite o nome do modelo na barra lateral.")
            else:
                st.session_state.messages.append({"role": "user", "content": pergunta})
                with st.spinner("Gerando resposta..."):
                    resposta = gerar_resposta(provedor, nome, modelo, chave, retriever, ferramentas)
                st.session_state.messages.append(resposta)
                st.rerun()

    with col2:
        st.subheader("Ações")
        st.write("Faça perguntas ao bot sobre Joinville.")

        if st.button("Limpar histórico"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        if erro_mcp == "sem_url":
            st.info("Previsão do tempo desativada: `MCP_URL` não configurada.")
        elif erro_mcp:
            st.warning("Servidor MCP indisponível — respondendo só com a base local.")
            if st.button("Tentar reconectar"):
                st.rerun()
        else:
            st.success(f"MCP conectado ({len(ferramentas)} ferramenta(s))")


if __name__ == "__main__":
    main()
