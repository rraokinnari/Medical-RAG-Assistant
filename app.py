import os

import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma

from ingest import process_pdf


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🧠 Medical AI Assistant")

st.write(
    "A document-grounded medical research assistant "
    "using Retrieval-Augmented Generation (RAG)."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("📄 Upload Medical Documents")


uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# Process uploaded PDFs
# --------------------------------------------------

if uploaded_files:

    st.sidebar.write(
        f"Uploaded {len(uploaded_files)} document(s)"
    )


    for uploaded_file in uploaded_files:

        # Save uploaded file temporarily
        temp_path = os.path.join(
            "documents",
            uploaded_file.name
        )


        # Create documents directory
        os.makedirs(
            "documents",
            exist_ok=True
        )


        # Save PDF
        with open(temp_path, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )


        # Process PDF
        with st.spinner(
            f"Processing {uploaded_file.name}..."
        ):
            pages, chunks, already_exists = process_pdf(
                temp_path
            )

        if already_exists:

            st.sidebar.info(
                f"{uploaded_file.name} is already indexed."
            )

        else:
            st.sidebar.success(
                f"{uploaded_file.name} processed "
                f"({pages} pages, {chunks} chunks)"
            )
        


# --------------------------------------------------
# Create embedding model
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    collection_name="medical_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# Create Gemini model
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


# --------------------------------------------------
# Question
# --------------------------------------------------
# --------------------------------------------------
# Display previous chat messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)


# --------------------------------------------------
# Process question
# --------------------------------------------------

if question:

    # --------------------------------------------------
    # Display user question
    # --------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # Save user message

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    # --------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------

    with st.spinner(
        "Searching documents..."
    ):

        results = vector_store.similarity_search(
            question,
            k=3
        )


    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context_parts = []

    for document in results:

        source = document.metadata.get(
            "source"
        )

        page = document.metadata.get(
            "page"
        )


        context_parts.append(

            f"Source: {source}, Page: {page}\n"
            f"{document.page_content}"

        )


    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""
You are a medical research document assistant.

Answer the user's question using ONLY the
information provided in the context below.

Do not use outside knowledge.

If the answer cannot be found in the
provided context, say:

"I could not find this information in
the uploaded documents."

Keep the answer clear and concise.

User question:
{question}

Context:
{context}
"""


    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    with st.spinner(
        "Generating answer..."
    ):
        response = llm.invoke(
            prompt
        )

        # Gemini may return structured content.
        # Convert it to plain text for Streamlit/chat history.

        if isinstance(response.content, str):

            answer = response.content

        else:

            answer = "".join(
                item.get("text", "")
                for item in response.content
                if isinstance(item, dict)
                and item.get("type") == "text"
            )


    # --------------------------------------------------
    # Display answer
    # --------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)


        # Sources

        st.markdown("**Sources**")


        seen_sources = set()


        for document in results:

            source = document.metadata.get(
                "source"
            )

            page = document.metadata.get(
                "page"
            )


            source_key = (
                source,
                page
            )


            if source_key not in seen_sources:

                st.markdown(
                    f"📄 **{source}** — Page {page}"
                )

                seen_sources.add(
                    source_key
                )


    # --------------------------------------------------
    # Save assistant response
    # --------------------------------------------------

    source_text = "\n\n**Sources**\n"

    for document in results:

        source = document.metadata.get(
            "source"
        )

        page = document.metadata.get(
            "page"
        )

        source_text += (
            f"- 📄 {source} — Page {page}\n"
        )


    st.session_state.messages.append({

        "role": "assistant",

        "content": answer + source_text

    })


