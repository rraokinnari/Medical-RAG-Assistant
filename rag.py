from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma


# Load environment variables
load_dotenv()


# --------------------------------------------------
# 1. Create embedding model
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    collection_name="medical_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 3. Create Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


# --------------------------------------------------
# 4. Ask user a question
# --------------------------------------------------

question = input("\nAsk a question about the document: ")


# --------------------------------------------------
# 5. Retrieve relevant chunks
# --------------------------------------------------

results = vector_store.similarity_search(
    question,
    k=3
)


# --------------------------------------------------
# 6. Combine retrieved chunks
# --------------------------------------------------

context_parts = []

for document in results:

    source = document.metadata.get("source")
    page = document.metadata.get("page")

    context_parts.append(
        f"Source: {source}, Page: {page}\n"
        f"{document.page_content}"
    )


context = "\n\n".join(context_parts)


# --------------------------------------------------
# 7. Create prompt
# --------------------------------------------------

prompt = f"""
You are a medical research document assistant.

Answer the user's question using ONLY the information
provided in the context below.

Do not use outside knowledge.

If the answer cannot be found in the context,
say:

"I could not find this information in the uploaded documents."

Keep the answer clear and concise.

User question:
{question}

Context:
{context}
"""


# --------------------------------------------------
# 8. Generate answer
# --------------------------------------------------

response = llm.invoke(prompt)


# --------------------------------------------------
# 9. Display answer
# --------------------------------------------------

print("\n\n========== ANSWER ==========")
print(response.content)


# --------------------------------------------------
# 10. Display sources
# --------------------------------------------------

print("\n\n========== SOURCES ==========")

for document in results:

    print(
        f"📄 {document.metadata.get('source')} "
        f"— Page {document.metadata.get('page')}"
    )