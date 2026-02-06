import streamlit as st
from langchain_community.llms import Ollama
import langchain_core.output_parsers as output_parsers
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import time
import base64
from pathlib import Path

# Função para converter imagem em base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return data


with open('.streamlit/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
	
# Adicionar imagem como background
img_path = "images/joinville-img.jpg"
if Path(img_path).exists():
    img_base64 = get_base64(img_path)
    st.markdown(
        f"""
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
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 0;
        }}
        .main {{
            position: relative;
            z-index: 1;
        }}
		
        </style>
        """,
        unsafe_allow_html=True
    )
	
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

embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="chatville"
)


st.set_page_config(page_title="Chat HUD", layout="wide")


st.title("Chat Ville — Chat Bot gui de joinville")  

with st.sidebar:
	st.header("Configuração")
	model = st.selectbox("Modelo (opcional)", ["gemma:7b", "cnmoro/mistral_7b_portuguese:q5_K_M"], index=1)

prompt = ChatPromptTemplate.from_messages([
	("system", "Você é um assistente amigável chamado Bot-Ville que responderá somente sobre perguntas relacionadas à cidade de Joinville/Santa Catarina. "
    "Seu objetivo é fornecer informações úteis e precisas sobre a cidade. Com base neste contexto:\n{context}\n\nResponda a pergunta de forma detalhada somente com os dados forncediso a você."),
	("human", "{input}")
])
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model=model)

chain = prompt | llm | output_parsers.StrOutputParser()

if "messages" not in st.session_state:
	st.session_state.messages = []
	
if "user_input" not in st.session_state:
    st.session_state.user_input = " "


def append_message(role, content):
	st.session_state.messages.append({"role": role, "content": content})


def get_response(user_text: str) -> str:
	try:
		# Monta histórico (opcional) -- não enviado ao retriever
		messages = []
		for m in st.session_state.messages:
			role = "Usuário" if m["role"] == "user" else "Assistente"
			messages.append(f"{role}: {m['content']}")

		# Recupera documentos relevantes e monta o contexto
		docs = retriever.invoke(user_text)
		context = "\n".join([d.page_content for d in docs])

		# Invoca a chain passando contexto e input do usuário
		response = chain.invoke({"context": context, "input": user_text})

		return response or "(sem resposta)"

	except Exception as e:
		return f"Erro ao chamar Ollama: {e}"

col1, col2 = st.columns([3, 1])

with col1:
	st.subheader("Conversa")
	chat_box = st.container()
	with chat_box:
		for msg in st.session_state.messages:
			if msg["role"] == "user":
				st.markdown(f"**Você:** {msg['content']}")
			else:
				st.markdown(f"**Bot:** {msg['content']}")

	user_input = st.text_input("Digite sua mensagem", key="user_input")
	if st.button("Enviar") or (user_input and st.session_state.get('auto_send')):
		if user_input:
			append_message("user", user_input)
			with st.spinner("Gerando resposta..."):
				reply = get_response(user_input)
			append_message("assistant", reply)
			st.rerun()
			# clear input
			# st.session_state.user_input = " "

	if st.button("Limpar histórico"):
		st.session_state.messages = []
		st.rerun()

with col2:
	st.subheader("Ações")
	st.write("Faça perguntas ao bot sobre Joinville.")
