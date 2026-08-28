import os

import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma

from ingest import process_pdf


# ==================================================
# Load environment variables
# ==================================================

load_dotenv()


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🧠",
    layout="wide"
)


# ==================================================
# Session state
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


# ==================================================
# Title
# ==================================================

st.title("🧠 Medical AI Assistant")

st.write(
    "A document-grounded medical research assistant "
    "using Retrieval-Augmented Generation (RAG)."
)


# ==================================================
# Sidebar - Upload documents
# ==================================================

st.sidebar.header("📄 Research Papers")


uploaded_files = st.sidebar.file_uploader(
    "Upload research papers",
    type=["pdf"],
    accept_multiple_files=True
)


# ==================================================
# Process uploaded PDFs
# ==================================================

if uploaded_files:

    os.makedirs("documents", exist_ok=True)

    for uploaded_file in uploaded_files:

        temp_path = os.path.join(
            "documents",
            uploaded_file.name
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

        # Store filename in session state
        if uploaded_file.name not in st.session_state.uploaded_documents:

            st.session_state.uploaded_documents.append(
                uploaded_file.name
            )

        if already_exists:

            st.sidebar.info(
                f"✓ {uploaded_file.name} already indexed."
            )

        else:

            st.sidebar.success(
                f"✓ {uploaded_file.name} processed "
                f"({pages} pages, {chunks} chunks)"
            )


# ==================================================
# Display uploaded papers
# ==================================================

if st.session_state.uploaded_documents:

    st.sidebar.subheader("📚 Available Papers")

    for document in st.session_state.uploaded_documents:

        st.sidebar.write(
            f"📄 {document}"
        )


# ==================================================
# Research Mode
# ==================================================

st.sidebar.subheader("🔬 Research Mode")


mode = st.sidebar.radio(
    "Choose how you want to use the assistant:",
    [
        "🔎 Research Q&A",
        "📊 Compare Papers"
    ]
)


# ==================================================
# Paper selection for comparison
# ==================================================

selected_papers = []


if mode == "📊 Compare Papers":

    if len(st.session_state.uploaded_documents) < 2:

        st.sidebar.warning(
            "Upload at least 2 research papers "
            "to use comparison mode."
        )

    else:

        selected_papers = st.sidebar.multiselect(
            "Select papers to compare:",
            st.session_state.uploaded_documents,
            help="Select 2 or more papers."
        )

        if len(selected_papers) >= 2:

            st.sidebar.success(
                f"{len(selected_papers)} papers selected."
            )

        elif len(selected_papers) == 1:

            st.sidebar.warning(
                "Please select at least 2 papers."
            )


# ==================================================
# Create embedding model
# ==================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ==================================================
# Connect to ChromaDB
# ==================================================

vector_store = Chroma(
    collection_name="medical_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ==================================================
# Gemini model
# ==================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


# ==================================================
# Display previous chat messages
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ==================================================
# Chat input
# ==================================================

if mode == "📊 Compare Papers":

    placeholder = (
        "Ask a question about the selected papers..."
    )

else:

    placeholder = (
        "Ask a question about your research papers..."
    )


question = st.chat_input(
    placeholder
)


# ==================================================
# Process question
# ==================================================

if question:

    # --------------------------------------------------
    # Validate comparison mode
    # --------------------------------------------------

    if mode == "📊 Compare Papers":

        if len(selected_papers) < 2:

            st.error(
                "Please select at least 2 papers "
                "before asking a comparison question."
            )

            st.stop()


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


    # ==================================================
    # Retrieve documents
    # ==================================================

    with st.spinner(
        "Searching research papers..."
    ):

        # --------------------------------------------------
        # Normal Research Q&A
        # --------------------------------------------------

        if mode == "🔎 Research Q&A":

            results = vector_store.similarity_search(
                question,
                k=5
            )


        # --------------------------------------------------
        # Compare Papers
        # --------------------------------------------------

        else:

            # ----------------------------------------------
            # Create the question embedding ONLY ONCE
            # ----------------------------------------------

            query_embedding = embeddings.embed_query(
                question
            )


            results = []


            # ----------------------------------------------
            # Search each selected paper separately
            # ----------------------------------------------

            for paper in selected_papers:

                paper_results = (
                    vector_store.similarity_search_by_vector(
                        query_embedding,
                        k=3,
                        filter={
                            "source": paper
                        }
                    )
                )


                results.extend(
                    paper_results
                )


    # ==================================================
    # Check retrieval
    # ==================================================

    if not results:

        st.error(
            "I could not find relevant information "
            "in the selected research papers."
        )

        st.stop()


    # ==================================================
    # Build context
    # ==================================================

    context_parts = []


    for document in results:

        source = document.metadata.get(
            "source"
        )

        page = document.metadata.get(
            "page"
        )


        context_parts.append(

            f"Research Paper: {source}\n"
            f"Page: {page}\n"
            f"Content:\n"
            f"{document.page_content}"

        )


    context = "\n\n--------------------\n\n".join(
        context_parts
    )


    # ==================================================
    # Create prompt
    # ==================================================

    if mode == "🔎 Research Q&A":

        prompt = f"""
You are an AI research assistant specialized in analyzing
scientific and medical research papers.

You must answer the user's question using ONLY the
information provided in the retrieved document context.

IMPORTANT RULES:

1. Do NOT use outside knowledge.
2. Do NOT invent facts, results, datasets, models, or citations.
3. If the required information is not present in the context,
   say:

"I could not find this information in the selected papers."

4. When stating information from a paper, include its source
and page number using this format:

(PaperName.pdf, Page X)

5. Keep the answer clear, concise, and useful for a researcher.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}

Now answer the user's question.
"""


    else:

        prompt = f"""
You are an AI research assistant specialized in comparing
scientific and medical research papers.

You must compare ONLY the papers selected by the user.

Use ONLY the information provided in the retrieved document
context below.

IMPORTANT RULES:

1. Do NOT use outside knowledge.
2. Do NOT invent facts, results, datasets, models, or citations.
3. Do NOT assume information that is not present in the context.
4. Clearly identify which information belongs to which paper.
5. Include the paper name and page number for important claims.
6. If information is unavailable for a paper, write:

"Not available in the retrieved context."

7. Do not compare papers that were not selected by the user.

For comparison questions, use a Markdown table whenever
appropriate.

Useful comparison categories include:

- Dataset
- Dataset size
- Problem addressed
- Methodology
- Model / Architecture
- Training approach
- Evaluation metrics
- Results
- Advantages
- Limitations

Only include categories supported by the retrieved context.

USER QUESTION:
{question}

SELECTED PAPER CONTEXT:
{context}

Now provide a clear comparison of the selected papers.
"""


    # ==================================================
    # Generate answer
    # ==================================================

    with st.spinner("Analyzing research papers..."):

        try:

            response = llm.invoke(prompt)

            # Gemini can return either plain text
            # or structured content.

            if isinstance(response.content, str):

                answer = response.content

            else:

                answer = "".join(
                    item.get("text", "")
                    for item in response.content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                )

            # Safety check
            if not answer.strip():

                answer = (
                    "I could not generate an answer from "
                    "the retrieved research paper content."
                )

        except Exception as e:

            st.error(
                f"Error while generating the answer: {e}"
            )

            st.stop()


    # ==================================================
    # Display answer
    # ==================================================

    with st.chat_message("assistant"):

        st.markdown(answer)

        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        st.markdown("**📚 Sources**")

        seen_sources = set()

        for document in results:

            source = document.metadata.get("source")
            page = document.metadata.get("page")

            source_key = (
                source,
                page
            )

            if source_key not in seen_sources:

                st.markdown(
                    f"📄 **{source}** — Page {page}"
                )

                seen_sources.add(source_key)


    # ==================================================
    # Save assistant response to chat history
    # ==================================================

    source_text = "\n\n**📚 Sources**\n"

    seen_sources = set()

    for document in results:

        source = document.metadata.get("source")
        page = document.metadata.get("page")

        source_key = (
            source,
            page
        )

        if source_key not in seen_sources:

            source_text += (
                f"- 📄 {source} — Page {page}\n"
            )

            seen_sources.add(source_key)


    st.session_state.messages.append({

        "role": "assistant",

        "content": answer + source_text

    })