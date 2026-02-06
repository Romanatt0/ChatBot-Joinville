from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


with open("texto.txt", "r", encoding="utf-8") as f:
    docs = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    
)

chunks_text = splitter.split_text(docs)


for i, c in enumerate(chunks_text):
    print(f"\n--- Chunk {i} ---")
    print(c)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe"
) 

vectorstore = Chroma.from_texts(
    texts=chunks_text,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="chatville"
)

vectorstore.persist()
