"""Avalia o Chat Ville com perguntas de gabarito.

Mede duas coisas:
- BUSCA: se o trecho com a resposta aparece entre os resultados da busca (rápido).
- RESPOSTAS: se a resposta final do agente contém o valor esperado (usa o LLM).

Uso:
    python avaliar.py                                    # só a busca
    python avaliar.py --provedor ollama --modelo qwen2.5:3b
    python avaliar.py --provedor groq --modelo llama-3.3-70b-versatile

Rode antes e depois de qualquer mudança (dados, divisão do texto, prompt, modelo)
para saber se ela realmente melhorou o bot.
"""
import argparse
import re
import time

from chatville import agente, banco, config, llm

PERGUNTAS = [
    ("Quando Joinville foi fundada?", "1851"),
    ("Qual a população de Joinville?", "616317"),
    ("Qual o gentílico de quem nasce em Joinville?", "joinvilense"),
    ("Qual era o primeiro nome de Joinville?", "dona francisca"),
    ("Quem é o prefeito de Joinville?", "adriano silva"),
    ("Qual a área do município de Joinville?", "1127,946"),
    ("Qual é o tipo de clima de Joinville?", "subtropical"),
    ("Qual o IDH de Joinville?", "0,809"),
    ("Quais municípios fazem divisa com Joinville?", "araquari"),
    ("Quantos imigrantes vieram na barca Colon?", "191"),
    ("Qual a distância de Joinville até a capital do estado?", "180km"),
    ("Quais são os apelidos de Joinville?", "cidade das flores"),
    ("Qual escola de teatro russa tem sede em Joinville?", "bolshoi"),
    ("Qual o principal time de futebol de Joinville?", "joinville esporte clube"),
    ("Qual o PIB de Joinville?", "30785682"),
]


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"(?<=\d)[\s.  ](?=\d)", "", s)  # 616 317 / 616.317 -> 616317
    return re.sub(r"\s+", " ", s).replace("180 km", "180km")


def main():
    parser = argparse.ArgumentParser(description="Avalia a busca e, opcionalmente, o agente.")
    parser.add_argument("--provedor", help="groq | openai | anthropic | ollama")
    parser.add_argument("--modelo", help="nome do modelo (padrão: o primeiro do modelos.toml)")
    parser.add_argument("--destino", choices=banco.DESTINOS, help="banco a avaliar (padrão: o do MODO)")
    args = parser.parse_args()

    vectorstore = banco.abrir_banco(banco.get_embeddings(), args.destino)
    aviso = banco.verificar_banco(vectorstore)
    if aviso:
        print("AVISO:", aviso)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    acertos = 0
    for q, esperado in PERGUNTAS:
        textos = norm(" ".join(d.page_content for d in retriever.invoke(q)))
        ok = esperado in textos
        acertos += ok
        print(f"{'✅' if ok else '❌'} busca | {q}")
    print(f"\nBUSCA: {acertos}/{len(PERGUNTAS)}")

    if args.provedor:
        modelo = args.modelo or config.provedores_disponiveis()[args.provedor]["modelos"][0]
        ferramentas, _ = agente.carregar_ferramentas_mcp(config.MCP_URL)
        ag, busca = agente.criar_agente(llm.get_llm(args.provedor, modelo), retriever, ferramentas)
        acertos, t0 = 0, time.time()
        for q, esperado in PERGUNTAS:
            try:
                resposta, _ = agente.responder(ag, busca, [{"role": "user", "content": q}])
            except Exception as e:
                resposta = llm.mensagem_de_erro(e, args.provedor)
            ok = esperado in norm(resposta)
            acertos += ok
            print(f"{'✅' if ok else '❌'} {modelo} | {q} -> {resposta[:110]!r}")
        print(f"\nRESPOSTAS ({args.provedor}/{modelo}): {acertos}/{len(PERGUNTAS)} em {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
