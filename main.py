from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = TextLoader("rag_db.md", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

print("RAG система готова")

while True:

    question = input("\nВведите вопрос: ")

    if question == "exit":
        break

    results = retriever.invoke(question)

    print("\nНайденный контекст:\n")

    for doc in results:
        print(doc.page_content)
        print("-----------")