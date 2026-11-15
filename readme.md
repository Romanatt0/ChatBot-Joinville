# Chat Ville — Chatbot sobre Joinville

Chat Ville é um projeto pessoal de estudo sobre **agentes de IA com LangChain**. É um chatbot em Streamlit, especializado na cidade de Joinville (SC), que roda **100% local** via [Ollama](https://ollama.com).

O bot é um **agente**: a cada pergunta, o modelo decide sozinho qual ferramenta usar — ou se deve recusar a pergunta — e responde com base no que as ferramentas retornarem:

| Ferramenta | Origem | Quando o agente usa |
|---|---|---|
| `buscar_info_joinville` | Local (RAG sobre `dados/joinville.md` no Chroma) | História, dados, cultura e turismo da cidade |
| `previsao_tempo_joinville` | Servidor [MCP](https://modelcontextprotocol.io) (API Open-Meteo) | Clima, temperatura e chuva |

O agente também recebe o **histórico da conversa**, então entende perguntas de continuação ("E quantos habitantes tem?"). A interface mostra quais ferramentas foram usadas em cada resposta.

## Arquitetura

```
┌──────────────┐  pergunta + histórico  ┌───────────────────────────┐
│  Streamlit   │ ─────────────────────► │  Agente LangChain         │
│  (app.py)    │ ◄───────────────────── │  (create_agent + Ollama)  │
└──────────────┘  resposta + ferramentas└──────┬─────────────┬──────┘
                                               │             │
                         buscar_info_joinville │             │ previsao_tempo_joinville
                                               ▼             ▼
                                 ┌──────────────────┐  ┌──────────────────────┐
                                 │ Chroma           │  │ Servidor MCP         │
                                 │ (chroma_db/)     │  │ localhost:8000/mcp   │
                                 │ gerado por rag.py│  │                      │
                                 └──────────────────┘  └──────────────────────┘
```

## Como funciona

**Base de conhecimento.** `dados/joinville.md` é um artigo sobre Joinville organizado em seções Markdown, com uma seção *Dados gerais* (fundação, população, prefeito, área...) em formato de lista. O `rag.py` divide o texto respeitando essas seções:

- cada item da lista *Dados gerais* vira um trecho próprio, então a busca acha exatamente o fato pedido;
- as demais seções são divididas em trechos de até 1000 caracteres, sem misturar seções;
- todo trecho começa com o caminho da seção (`[Joinville > História > Origens e povoamento]`), o que dá contexto à busca e ao modelo.

**Busca de reserva.** Modelos pequenos às vezes respondem "de cabeça", sem consultar nada (e o Ollama não permite forçar o uso de ferramentas). Quando isso acontece e a resposta não é uma recusa, o app faz a busca pelo agente com a própria pergunta e pede a resposta de novo. A interface marca esses casos como `buscar_info_joinville (automática)`.

**Limite de resposta.** As respostas são limitadas a 1024 tokens (`num_predict`): sem isso, modelos pequenos às vezes entram em loop repetindo texto e a resposta nunca termina.

## Avaliação

`avaliar.py` faz 15 perguntas com resposta conhecida e mede:

- **Busca:** se o trecho com a resposta aparece nos resultados;
- **Respostas:** se a resposta final do agente contém o valor correto.

| Versão | Busca | Respostas (`qwen2.5:3b`) |
|---|---|---|
| Texto cru da Wikipédia, contexto sempre no prompt | 9/15 | 4/15 |
| Texto organizado em seções + fatos isolados | 14/15 | 8/15 |
| + busca de reserva | 14/15 | 12/15 |

```powershell
python avaliar.py                      # só a busca (segundos)
python avaliar.py --agente qwen2.5:3b  # busca + respostas (~1,5 min)
```

**Limitações conhecidas** com o `qwen2.5:3b`: às vezes acrescenta informações que não estão na base (ex.: sobrenome do prefeito), nem sempre recusa perguntas fora do tema, e a busca não liga "apelidos" a "títulos" (o texto diz *"ostenta os títulos de..."*).

## Estrutura do projeto

| Arquivo | Função |
|---|---|
| `app.py` | Interface Streamlit e agente (RAG + ferramentas MCP) |
| `rag.py` | Gera a base vetorial a partir de `dados/joinville.md` |
| `avaliar.py` | Mede a taxa de acerto da busca e do agente com perguntas de gabarito |
| `config.py` | Modelos, caminhos, URL do MCP e prompt do bot |
| `dados/joinville.md` | Base de conhecimento sobre Joinville (Markdown com seções) |
| `.streamlit/` | Tema e CSS da interface |
| `images/` | Imagem de fundo |

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado e rodando
- O servidor MCP: **[Romanatt0/MCP-SERVER](https://github.com/Romanatt0/MCP-SERVER)**

## Como rodar

### 1. Ambiente e dependências

```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Modelos do Ollama

```powershell
ollama pull nomic-embed-text-v2-moe   # embeddings (obrigatório)
ollama pull qwen2.5:3b                # modelo de chat padrão
```

Opcional: `ollama pull llama3.1:8b` — mais lento, mas segue melhor as instruções e usa as ferramentas com mais consistência.

> O modelo de chat **precisa suportar ferramentas** (tool calling). Confira com `ollama show <modelo>` — deve aparecer `tools` em *Capabilities*. Modelos como `gemma:7b` não suportam e dão erro.

### 3. Gerar a base vetorial

```powershell
python rag.py
```

Rode de novo sempre que alterar `dados/joinville.md` ou o modelo de embeddings. A pasta `chroma_db/` é gerada localmente e não é versionada.

### 4. Subir o servidor MCP

> ⚠️ **O Chat Ville precisa ser apontado para um servidor MCP.** É ele que fornece as ferramentas do agente (como a previsão do tempo). Sem o servidor, o chat ainda abre e responde com a base local, mas exibe o aviso *"Servidor MCP indisponível"* e não consegue responder perguntas sobre o clima.

Em outro terminal, clone e rode o servidor:

```powershell
git clone https://github.com/Romanatt0/MCP-SERVER.git
cd MCP-SERVER
pip install fastmcp requests
cd src
python my_server.py
```

Por padrão o servidor sobe em `http://localhost:8000/mcp`, que é o endereço que o Chat Ville usa.

**Usando outro endereço:** se o servidor MCP estiver em outra porta ou máquina, aponte para ele pela variável de ambiente `CHATVILLE_MCP_URL` antes de iniciar o app:

```powershell
$env:CHATVILLE_MCP_URL = "http://meu-servidor:9000/mcp"
```

Se o servidor cair ou subir depois do app, use o botão **"Tentar reconectar"** na coluna *Ações*.

### 5. Iniciar o app

```powershell
py -m streamlit run app.py
```

Acesse `http://localhost:8501`. Quando tudo estiver certo, a coluna *Ações* mostra **"MCP conectado (1 ferramenta(s))"**.

## Configuração

Todas as opções ficam em `config.py` e podem ser sobrescritas por variáveis de ambiente:

| Variável | Padrão | Descrição |
|---|---|---|
| `CHATVILLE_MCP_URL` | `http://localhost:8000/mcp` | Endereço do servidor MCP |
| `CHATVILLE_EMBEDDING_MODEL` | `nomic-embed-text-v2-moe` | Modelo de embeddings do Ollama |
| `CHATVILLE_CHROMA_DIR` | `./chroma_db` | Pasta da base vetorial |
| `CHATVILLE_COLLECTION` | `chatville` | Nome da coleção no Chroma |
| `CHATVILLE_RETRIEVER_K` | `3` | Quantidade de trechos retornados pela busca |
| `CHATVILLE_HISTORY_LIMIT` | `10` | Mensagens anteriores enviadas ao agente (memória) |
| `CHATVILLE_SOURCE_TEXT` | `dados/joinville.md` | Texto usado pelo `rag.py` |

## Exemplos de perguntas

- "Qual a origem do nome Joinville?"
- "Quais são os apelidos da cidade?"
- "Como está o tempo em Joinville agora?" *(usa a ferramenta MCP)*

## Problemas comuns

| Sintoma | Causa provável |
|---|---|
| Aviso "Servidor MCP indisponível" | Servidor MCP não está rodando ou `CHATVILLE_MCP_URL` está errada |
| Respostas sem relação com Joinville / sem contexto | `chroma_db/` não foi gerado — rode `python rag.py` |
| Erro `model not found` | Modelo não baixado — rode `ollama pull <modelo>` |
| Erro `does not support tools` | O modelo escolhido não suporta ferramentas — use um da lista |
| Erro de conexão com Ollama | Ollama não está rodando (`ollama serve`) |

## Próximos passos

- Mais ferramentas no servidor MCP (eventos, pontos turísticos, transporte)
- Ampliar a base de conhecimento e mostrar as fontes nas respostas
- Ampliar o conjunto de avaliação (perguntas de continuação e fora do tema)
- Interface com `st.chat_message` e respostas em streaming

## Contato

Projeto pessoal — sugestões e problemas são bem-vindos nas issues do GitHub.
