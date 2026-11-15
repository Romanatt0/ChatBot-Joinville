# Chat Ville — Chatbot sobre Joinville

Chat Ville é um projeto pessoal de estudo sobre **agentes de IA com LangChain**: um chatbot em Streamlit, especializado na cidade de Joinville (SC), que roda **na sua máquina** (Ollama + Chroma) ou **online** no Streamlit Community Cloud (Groq/OpenAI/Anthropic + Qdrant Cloud).

O bot é um **agente**: a cada pergunta, o modelo decide sozinho qual ferramenta usar — ou se deve recusar a pergunta — e responde com base no que as ferramentas retornarem:

| Ferramenta | Origem | Quando o agente usa |
|---|---|---|
| `buscar_info_joinville` | Banco vetorial (Chroma ou Qdrant) com `dados/joinville.md` | História, dados, cultura e turismo da cidade |
| `previsao_tempo_joinville` | Servidor [MCP](https://modelcontextprotocol.io) (API Open-Meteo) | Clima, temperatura e chuva |

O agente também recebe o **histórico da conversa** (entende "E quantos habitantes tem?") e a interface mostra quais ferramentas foram usadas em cada resposta.

## Modos de execução

| | `MODO=local` | `MODO=cloud` |
|---|---|---|
| Banco vetorial | Chroma (pasta `chroma_db/`) | Qdrant Cloud |
| Provedores de LLM | Groq, OpenAI, Anthropic, **Ollama** | Groq (padrão), OpenAI, Anthropic |
| Configuração | arquivo `.env` | `st.secrets` (painel do Streamlit Cloud) |
| Dependências | `requirements-local.txt` | `requirements.txt` |

Nos dois modos, o usuário escolhe na barra lateral o **provedor**, o **modelo** (ou digita um personalizado) e, se quiser, informa a **própria chave de API**.

## Arquitetura

```
┌──────────────┐  pergunta + histórico  ┌─────────────────────────────┐
│  Streamlit   │ ─────────────────────► │  Agente LangChain           │
│  (app.py)    │ ◄───────────────────── │  create_agent + get_llm()   │
└──────────────┘  resposta + ferramentas│  Groq│OpenAI│Anthropic│Ollama│
                                        └──────┬───────────────┬──────┘
                         buscar_info_joinville │               │ previsao_tempo_joinville
                                               ▼               ▼
                              ┌─────────────────────────┐  ┌──────────────────┐
                              │ Banco vetorial          │  │ Servidor MCP     │
                              │ local: Chroma           │  │ (MCP_URL)        │
                              │ cloud: Qdrant Cloud     │  └──────────────────┘
                              │ embeddings: FastEmbed   │
                              └────────────▲────────────┘
                                           │ grava (só o ingestao.py)
                                  dados/joinville.md
```

## Estrutura do projeto

| Arquivo | Função |
|---|---|
| `app.py` | Interface Streamlit (barra lateral, chat, mensagens de erro) |
| `chatville/config.py` | Lê `MODO` e as configurações do `.env` / `st.secrets` |
| `chatville/llm.py` | `get_llm(provedor, modelo, api_key)` e tradução de erros dos provedores |
| `chatville/banco.py` | Embeddings (FastEmbed) e banco vetorial (Chroma ou Qdrant) |
| `chatville/agente.py` | Ferramentas, MCP, criação do agente e lógica de resposta |
| `modelos.toml` | **Lista editável** de provedores e modelos da barra lateral |
| `ingestao.py` | Cria o banco vetorial (`--destino chroma\|qdrant\|ambos`) |
| `avaliar.py` | Mede a taxa de acerto da busca e do agente |
| `dados/joinville.md` | Base de conhecimento (Markdown com seções) |
| `.env.example` | Modelo de configuração do modo local |
| `.streamlit/secrets.toml.example` | Modelo dos secrets do modo cloud |

---

## Rodar localmente (`MODO=local`)

### 1. Ambiente e dependências

```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements-local.txt
```

### 2. Configuração

```powershell
copy .env.example .env
```

O `.env` já vem com `MODO=local`. As chaves de API são opcionais: dá para digitá-las na barra lateral, ou preenchê-las no `.env` para não precisar digitar toda vez.

### 3. Criar o banco vetorial

```powershell
python ingestao.py --destino chroma
```

Na primeira vez, o modelo de embeddings (~220 MB) é baixado automaticamente.

### 4. (Opcional) Modelos do Ollama

Para conversar sem API e sem internet, instale o [Ollama](https://ollama.com) e baixe um modelo **com suporte a ferramentas**:

```powershell
ollama pull qwen2.5:3b     # leve
ollama pull llama3.1:8b    # mais lento, segue melhor as instruções
```

### 5. (Opcional) Servidor MCP — previsão do tempo

> ⚠️ **A ferramenta de clima depende de um servidor MCP.** Sem ele, o chat funciona normalmente, mas não responde sobre o clima.

Em outro terminal:

```powershell
git clone https://github.com/Romanatt0/MCP-SERVER.git
cd MCP-SERVER
pip install fastmcp requests
cd src
python my_server.py
```

Ele sobe em `http://localhost:8000/mcp`, o valor padrão de `MCP_URL` no `.env`. Se o servidor subir depois do app, use o botão **"Tentar reconectar"**.

### 6. Iniciar

```powershell
streamlit run app.py
```

---

## Rodar o modo cloud na sua máquina (`MODO=cloud`)

Útil para testar antes do deploy.

1. Crie um cluster gratuito no [Qdrant Cloud](https://cloud.qdrant.io) e copie a **URL** e a **API key** do cluster.
2. Crie uma chave gratuita no [Groq](https://console.groq.com/keys).
3. Crie o arquivo de secrets a partir do modelo e preencha:
   ```powershell
   copy .streamlit\secrets.toml.example .streamlit\secrets.toml
   ```
   > O `.streamlit/secrets.toml` está no `.gitignore` e **nunca** deve ir para o git.
4. Envie a base de conhecimento para o Qdrant (o `ingestao.py` também lê o `secrets.toml`):
   ```powershell
   python ingestao.py --destino qdrant
   ```
5. Inicie: `streamlit run app.py`. A barra lateral deve mostrar **Modo: cloud**, sem o Ollama.

> Se existir um `.env` com `MODO=local`, ele tem prioridade sobre o `secrets.toml`. Remova a linha ou use `MODO=cloud`.

---

## Deploy no Streamlit Community Cloud

1. **Banco:** faça os passos 1, 2 e 4 da seção anterior. O Qdrant Cloud precisa ter a coleção `chatville` criada pelo `ingestao.py`. O app só consulta o banco, nunca grava nele.
2. **Código:** suba o projeto para o GitHub. Confira que `.env` e `.streamlit/secrets.toml` **não** foram enviados.
3. Em [share.streamlit.io](https://share.streamlit.io), clique em **Create app** e escolha o repositório, a branch e o arquivo principal `app.py`.
4. Em **Advanced settings**:
   - **Python version:** 3.12;
   - **Secrets:** cole o conteúdo do seu `secrets.toml` preenchido:
     ```toml
     MODO = "cloud"
     GROQ_API_KEY = "gsk_..."
     QDRANT_URL = "https://xxxxxxxx.cloud.qdrant.io"
     QDRANT_API_KEY = "..."
     # MCP_URL = "https://seu-servidor-mcp.exemplo.com/mcp"   # opcional
     ```
5. Clique em **Deploy**. O Streamlit Cloud instala o `requirements.txt`, sem Ollama nem Chroma. A primeira abertura demora um pouco mais, por causa do download do modelo de embeddings.

**Sobre as chaves no cloud**

- A `GROQ_API_KEY` dos secrets é usada por **qualquer visitante** que não informe a própria chave. Quando a cota acabar, o app mostra a mensagem de limite atingido e sugere que o visitante use a chave dele.
- Chaves de OpenAI e Anthropic **não** são lidas dos secrets no modo cloud: cada usuário informa a sua na barra lateral.
- As chaves digitadas ficam só na sessão do navegador (`st.session_state`): não são salvas, registradas em log nem exibidas.

**Previsão do tempo no cloud:** o Streamlit Cloud não enxerga o `localhost` da sua máquina. Para ter a ferramenta de clima online, publique o [MCP-SERVER](https://github.com/Romanatt0/MCP-SERVER) em um serviço que rode Python com URL pública e configure `MCP_URL` nos secrets. Sem `MCP_URL`, o app mostra "Previsão do tempo desativada" e segue respondendo com a base.

---

## Configuração

Ordem de leitura: variável de ambiente (`.env`) → `st.secrets` → padrão.

| Variável | Padrão | Descrição |
|---|---|---|
| `MODO` | `local` | `local` ou `cloud` |
| `GROQ_API_KEY` | — | Chave do app para o Groq (local e cloud) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Chaves do app, **só no modo local** |
| `QDRANT_URL` / `QDRANT_API_KEY` | — | Cluster do Qdrant Cloud |
| `MCP_URL` | `http://localhost:8000/mcp` (local) / vazio (cloud) | Servidor MCP |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo do FastEmbed |
| `COLLECTION_NAME` | `chatville` | Coleção no Chroma/Qdrant |
| `RETRIEVER_K` | `3` | Trechos retornados por busca |
| `HISTORY_LIMIT` | `10` | Mensagens anteriores enviadas ao agente |
| `MAX_TOKENS` | `1024` | Tamanho máximo da resposta |

### Modelos (`modelos.toml`)

A lista de provedores e modelos da barra lateral fica em `modelos.toml`. Edite à vontade: o primeiro modelo de cada provedor é o padrão. A opção **"Outro (digitar)…"** permite usar qualquer modelo. O modelo **precisa suportar ferramentas** (tool calling).

### Embeddings

O embedding pertence ao **banco**, não ao LLM: trocar de provedor de chat não muda nada na busca. Os dois modos usam o FastEmbed (roda na CPU, sem API).

> ⚠️ **Se mudar `EMBEDDING_MODEL`, recrie o banco** com `python ingestao.py`. Cada trecho guarda o nome do modelo usado. Se o banco tiver sido criado com outro, o app avisa na tela.

---

## Como funciona

**Base de conhecimento.** `dados/joinville.md` é organizado em seções Markdown, com uma seção *Dados gerais* em formato de lista. O `ingestao.py` divide o texto respeitando as seções: cada item da lista vira um trecho próprio, e todo trecho começa com o caminho da seção (`[Joinville > História > ...]`).

**Busca de reserva.** Alguns modelos respondem "de cabeça", sem consultar nada. Quando isso acontece e a resposta não é uma recusa, o app faz a busca pelo agente e pede a resposta de novo. A interface marca esses casos como `buscar_info_joinville (automática)`.

**Erros amigáveis.** Chave inválida, limite de uso (429), modelo inexistente e falha de conexão viram mensagens claras. O texto original do erro nunca é exibido, porque alguns provedores incluem parte da chave nele.

## Avaliação

`avaliar.py` faz 15 perguntas com resposta conhecida e mede se a **busca** encontra o trecho certo e se a **resposta** do agente contém o valor correto.

```powershell
python avaliar.py                                          # só a busca
python avaliar.py --provedor ollama --modelo qwen2.5:3b    # busca + agente
python avaliar.py --provedor groq                          # usa o 1º modelo do modelos.toml
```

| Versão | Busca | Respostas (`qwen2.5:3b`) |
|---|---|---|
| Texto cru da Wikipédia, contexto sempre no prompt | 9/15 | 4/15 |
| Texto organizado em seções + fatos isolados (`nomic-embed-text`) | 14/15 | 8/15 |
| + busca de reserva (`nomic-embed-text`) | 14/15 | 12/15 |
| FastEmbed multilíngue (atual, local e cloud) | 12/15 | 13/15 |

O embedding atual (MiniLM multilíngue, 220 MB) busca um pouco pior que o `nomic-embed-text` do Ollama. Ele tem dificuldade com fatos curtos como "Prefeito: ..." e "IDH: ...". Em troca, roda no Streamlit Cloud e é o mesmo nos dois modos. O agente compensa parte da diferença refazendo a busca com outras palavras.

**Limitações conhecidas:** com modelos pequenos (`qwen2.5:3b`), às vezes acrescenta informações que não estão na base e nem sempre recusa perguntas fora do tema.

## Problemas comuns

| Sintoma | Causa provável |
|---|---|
| "Chave de API inválida" | Chave errada ou de outro provedor |
| "Limite de uso atingido" | Cota do provedor esgotada (429). No cloud, informe sua própria chave |
| "Banco vetorial vazio" / "criado com o embedding X" | Rode `python ingestao.py` (com o `--destino` certo) |
| "Não consegui abrir o banco vetorial" (cloud) | `QDRANT_URL` / `QDRANT_API_KEY` ausentes ou errados |
| "Servidor MCP indisponível" | Servidor MCP parado ou `MCP_URL` errada |
| "Previsão do tempo desativada" (cloud) | `MCP_URL` não configurada nos secrets |
| "O modelo escolhido não suporta ferramentas" | Escolha um modelo com tool calling |
| "Não consegui conectar ao Ollama" | Ollama não está rodando (`ollama serve`) |
| No Windows, aviso `WinError 1314` ao baixar o embedding | Falta permissão para criar links simbólicos. O FastEmbed baixa de outra fonte e funciona; para sumir o aviso, ative o *Modo de desenvolvedor* do Windows |

## Próximos passos

- Busca híbrida (palavras-chave + vetores) para melhorar fatos curtos como "prefeito" e "IDH"
- Publicar o servidor MCP para a previsão do tempo funcionar no cloud
- Mais ferramentas no servidor MCP (eventos, pontos turísticos, transporte)
- Interface com `st.chat_message` e respostas em streaming

## Contato

Projeto pessoal — sugestões e problemas são bem-vindos nas issues do GitHub.
