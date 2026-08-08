from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Load environment variables
load_dotenv()


# --------------------------------------------------
# 1. PDF LOCATION
# --------------------------------------------------

PDF_PATH = Path("documents/RevisedSTROKE-XDL.pdf")


# --------------------------------------------------
# 2. READ PDF
# --------------------------------------------------

reader = PdfReader(PDF_PATH)

pages = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if text:
        pages.append({
            "page_number": page_number,
            "text": text
        })


print(f"Number of pages: {len(pages)}")


# --------------------------------------------------
# 3. SPLIT TEXT INTO CHUNKS
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


chunks = []

for page in pages:

    page_chunks = text_splitter.split_text(page["text"])

    for chunk in page_chunks:

        chunks.append({
            "text": chunk,
            "page_number": page["page_number"]
        })


print(f"Number of chunks: {len(chunks)}")


# --------------------------------------------------
# 4. CREATE EMBEDDING MODEL
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# --------------------------------------------------
# 5. PREPARE TEXT + METADATA
# --------------------------------------------------

texts = [chunk["text"] for chunk in chunks]

metadatas = [
    {
        "source": PDF_PATH.name,
        "page": chunk["page_number"]
    }
    for chunk in chunks
]


# --------------------------------------------------
# 6. STORE IN CHROMADB
# --------------------------------------------------

vector_store = Chroma(
    collection_name="medical_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


vector_store.add_texts(
    texts=texts,
    metadatas=metadatas
)


print("Successfully stored chunks in ChromaDB!")