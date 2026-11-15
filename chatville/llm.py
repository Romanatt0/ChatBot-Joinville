"""Factory de modelos de chat e tradução de erros dos provedores.

O resto do app só chama get_llm() e mensagem_de_erro(): não sabe (nem precisa
saber) qual provedor está por trás.
"""
from chatville import config

# Variável de ambiente / secret com a chave "do app" de cada provedor.
CHAVES_DO_APP = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ChaveAusente(Exception):
    """O provedor exige chave de API e nenhuma foi informada."""


def precisa_de_chave(provedor: str) -> bool:
    return provedor in CHAVES_DO_APP


def chave_do_app(provedor: str) -> str | None:
    """Chave configurada pelo dono do app (.env ou st.secrets).

    - Groq: usada em qualquer modo quando o usuário não informa a dele.
    - OpenAI/Anthropic: só no modo local (na sua máquina). No cloud, uma chave
      desses provedores nos secrets ficaria disponível para qualquer visitante.
    """
    if provedor not in CHAVES_DO_APP:
        return None
    if provedor == "groq" or config.MODO == "local":
        return config.ler(CHAVES_DO_APP[provedor])
    return None


def get_llm(provedor: str, modelo: str, api_key: str | None = None):
    """Devolve um chat model do LangChain pronto para o agente.

    `api_key` é a chave digitada pelo usuário; se vier vazia, usa a chave do app
    (quando permitido). Os imports são feitos aqui dentro para que o modo cloud
    não precise instalar o pacote do Ollama.
    """
    chave = api_key or chave_do_app(provedor)
    if precisa_de_chave(provedor) and not chave:
        raise ChaveAusente(provedor)

    comum = {"model": modelo, "max_tokens": config.MAX_TOKENS,
             "timeout": config.TIMEOUT_SEGUNDOS, "max_retries": 1}

    if provedor == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=chave, temperature=0, **comum)

    if provedor == "openai":
        from langchain_openai import ChatOpenAI
        # Sem temperature: alguns modelos recentes da OpenAI só aceitam o valor padrão.
        return ChatOpenAI(api_key=chave, **comum)

    if provedor == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(api_key=chave, temperature=0, **comum)

    if provedor == "ollama":
        if config.MODO != "local":
            raise ValueError("O Ollama só está disponível no modo local.")
        from langchain_ollama import ChatOllama
        # num_predict limita a resposta: sem ele, modelos pequenos às vezes
        # entram em loop repetindo texto e a resposta nunca termina.
        return ChatOllama(model=modelo, temperature=0, num_predict=config.MAX_TOKENS)

    raise ValueError(f"Provedor desconhecido: {provedor!r}")


# ---------------------------------------------------------------------------
# Erros
# ---------------------------------------------------------------------------

def _cadeia(exc: BaseException):
    """A exceção, suas causas e, para ExceptionGroup, as sub-exceções."""
    vistos, pilha = set(), [exc]
    while pilha:
        e = pilha.pop()
        if e is None or id(e) in vistos:
            continue
        vistos.add(id(e))
        yield e
        pilha += [e.__cause__, e.__context__, *getattr(e, "exceptions", ())]


def _status(e: BaseException) -> int | None:
    status = getattr(e, "status_code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def mensagem_de_erro(exc: BaseException, provedor: str, nome_provedor: str | None = None) -> str:
    """Traduz um erro do provedor numa mensagem amigável.

    Nunca devolve o texto original do erro: algumas mensagens dos provedores
    incluem parte da chave de API.
    """
    nome = nome_provedor or provedor
    for e in _cadeia(exc):
        tipo = type(e).__name__
        status = _status(e)
        texto = str(e).lower()  # usado só para classificar, nunca exibido

        if isinstance(e, ChaveAusente):
            return f"🔑 Informe sua chave de API do {nome} na barra lateral."
        if status in (401, 403) or "Authentication" in tipo or "PermissionDenied" in tipo:
            return f"🔑 Chave de API inválida para o {nome}. Confira a chave na barra lateral."
        if status == 429 or "RateLimit" in tipo:
            dica = " ou informe sua própria chave" if provedor == "groq" else ""
            return f"⏳ Limite de uso do {nome} atingido. Aguarde um pouco{dica} e tente de novo."
        if "does not support tools" in texto:
            return f"🧰 O modelo escolhido não suporta ferramentas. Escolha outro modelo do {nome}."
        if status == 404 or "NotFound" in tipo:
            return f"❓ Modelo não encontrado no {nome}. Confira o nome do modelo."
        if (isinstance(e, (ConnectionError, TimeoutError))
                or any(p in tipo for p in ("Connect", "Timeout", "NetworkError"))):
            if provedor == "ollama":
                return "🔌 Não consegui conectar ao Ollama. Ele está rodando (`ollama serve`)?"
            return f"🔌 Não consegui conectar ao {nome}. Verifique sua internet e tente de novo."

    return f"⚠️ Erro inesperado ao gerar a resposta ({type(exc).__name__})."
