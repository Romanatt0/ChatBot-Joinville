"""Avalia o Chat Ville com perguntas de gabarito.

Mede duas coisas:
- BUSCA: se o trecho com a resposta aparece entre os resultados da busca (rápido).
- RESPOSTAS: se a resposta final do agente contém o valor esperado (usa o LLM).

Uso:
    python avaliar.py                      # só a busca
    python avaliar.py --agente qwen2.5:3b  # busca + respostas do agente

Rode antes e depois de qualquer mudança (dados, divisão do texto, prompt, modelo)
para saber se ela realmente melhorou o bot.
"""
import re
import sys
import time

import app

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
    retriever = app.load_retriever()
    acertos = 0
    for q, esperado in PERGUNTAS:
        textos = norm(" ".join(d.page_content for d in retriever.invoke(q)))
        ok = esperado in textos
        acertos += ok
        print(f"{'✅' if ok else '❌'} busca | {q}")
    print(f"\nBUSCA: {acertos}/{len(PERGUNTAS)}")

    if "--agente" in sys.argv:
        model = sys.argv[sys.argv.index("--agente") + 1]
        tools, _ = app.get_mcp_tools()
        agent, rag_tool = app.load_agent(model, tuple(t.name for t in tools), tools, retriever)
        acertos, t0 = 0, time.time()
        for q, esperado in PERGUNTAS:
            reply, used = app.answer(agent, rag_tool, [{"role": "user", "content": q}])
            ok = esperado in norm(reply)
            acertos += ok
            print(f"{'✅' if ok else '❌'} {model} | {q} -> {reply[:110]!r}")
        print(f"\nRESPOSTAS ({model}): {acertos}/{len(PERGUNTAS)} em {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
