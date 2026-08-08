from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Load API key
load_dotenv()


# --------------------------------------------------
# 1. Create embedding model
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# --------------------------------------------------
# 2. Connect to existing ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    collection_name="medical_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 3. Ask a question
# --------------------------------------------------

question = input("\nAsk a question about the document: ")


# --------------------------------------------------
# 4. Perform similarity search
# --------------------------------------------------

results = vector_store.similarity_search(
    question,
    k=3
)


# --------------------------------------------------
# 5. Display retrieved chunks
# --------------------------------------------------

print("\n\n========== RETRIEVED DOCUMENTS ==========")

for i, document in enumerate(results, start=1):

    print(f"\n---------- Result {i} ----------")

    print("Source:", document.metadata.get("source"))
    print("Page:", document.metadata.get("page"))

    print("\nContent:")
    print(document.page_content)