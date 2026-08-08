import hashlib
from pathlib import Path

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma


def get_file_hash(pdf_path):
    """Create a unique hash for the PDF."""

    with open(pdf_path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def process_pdf(pdf_path):
    """
    Extract PDF text, split it into chunks,
    generate embeddings and store them in ChromaDB.
    """

    # --------------------------------------------------
    # 1. Generate unique document ID
    # --------------------------------------------------

    document_id = get_file_hash(pdf_path)

    filename = Path(pdf_path).name


    # --------------------------------------------------
    # 2. Create embedding model
    # --------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )


    # --------------------------------------------------
    # 3. Connect to ChromaDB
    # --------------------------------------------------

    vector_store = Chroma(
        collection_name="medical_documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )


    # --------------------------------------------------
    # 4. Check whether document already exists
    # --------------------------------------------------

    existing = vector_store.get(
        where={
            "document_id": document_id
        },
        limit=1
    )


    if existing["ids"]:

        return 0, 0, True


    # --------------------------------------------------
    # 5. Read PDF
    # --------------------------------------------------

    reader = PdfReader(pdf_path)

    pages = []


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages.append({
                "page_number": page_number,
                "text": text
            })


    # --------------------------------------------------
    # 6. Split text
    # --------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )


    texts = []
    metadatas = []


    for page in pages:

        chunks = text_splitter.split_text(
            page["text"]
        )


        for chunk_index, chunk in enumerate(chunks):

            texts.append(chunk)

            metadatas.append({

                "source": filename,

                "page": page["page_number"],

                "document_id": document_id,

                "chunk_id": chunk_index

            })


    # --------------------------------------------------
    # 7. Store in ChromaDB
    # --------------------------------------------------

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas
    )


    return len(pages), len(texts), False