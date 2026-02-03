import streamlit as st
from langchain_community.llms import Ollama
import langchain_core.output_parsers as output_parsers
from langchain_core.prompts import ChatPromptTemplate
import time

st.set_page_config(page_title="Chat HUD", layout="wide")
st.title("Chat Ville — Chat Bot gui de joinville")  

with st.sidebar:
	st.header("Configuração")
	model = st.selectbox("Modelo (opcional)", ["gemma:7b", "cnmoro/mistral_7b_portuguese:q5_K_M"], index=1)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente amigável chamado Bot-Ville que respondera somente sobre perguntas relacionadas a cidade de Joinville/Santa Catarina, seu objetivo é fornecer informações úteis e precisas sobre a cidade."),
    ("human", "{input}")
])

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
        messages = []

        for m in st.session_state.messages:
            role = "Usuário" if m["role"] == "user" else "Assistente"
            messages.append(f"{role}: {m['content']}")

        messages.append(f"Usuário: {user_text}")

        prompt = "\n".join(messages)

        response = chain.invoke(prompt)

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

