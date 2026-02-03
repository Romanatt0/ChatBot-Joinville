# Chat Ville — Chatbot sobre Joinville

Bem-vindo ao Chat Ville — um projeto pessoal sobre meu aprendizado de agentes de IA usando LangChain. Este repositório contém uma interface Streamlit simples que conecta um modelo local (por exemplo, via Ollama) e permite conversar com um chatbot especializado em informações sobre a cidade de Joinville, Santa Catarina.

Principais objetivos
- Documentar e experimentar conceitos de agentes e pipelines com LangChain.
- Construir um chatbot focado em responder perguntas sobre Joinville (história, pontos turísticos, serviços, eventos locais, etc.).
- Ter uma interface leve (HUD) em Streamlit para testes e demonstrações.

Como usar
1. Crie e ative um ambiente virtual (recomendado):
```powershell
python -m venv venv
.\venv\Scripts\Activate
```
2. Instale dependências (exemplo):
```powershell
pip install -r requirements.txt
```
Se você não tiver `requirements.txt`, instale pelo menos `streamlit` e as bibliotecas que usa com LangChain (por exemplo `langchain`, `langchain-community`, `ollama`), ou adapte à sua pilha.

3. Execute a app:
```powershell
py -m streamlit run app.py
```

Notas técnicas
- A UI principal está em `app.py` e usa Streamlit para exibir o histórico de mensagens e um campo de entrada.
- O projeto demonstra integração com LangChain (prompt templates, chains) e um LLM local (via `Ollama`), mas pode ser adaptado para usar OpenAI ou outro provedor.
- O `.gitignore` já ignora ambientes virtuais e caches.

Contribuições e melhorias que planejo
- Adicionar testes básicos e um `requirements.txt` explícito.
- Melhorar tratamento de contexto e persistência de histórico (ex: salvar conversas localmente).
- Documentar como configurar Ollama / modelos locais ou chaves de API para provedores remotos.

Contato
- Projeto pessoal — use issues no GitHub se quiser sugerir melhorias ou relatar problemas.

