import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import streamlit as st

api_key = os.getenv("OPENAI_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pgx_docs")
DOCS_PATH = os.getenv("DOCS_PATH", "TPMT_doc_cleaned.txt")
EMBEDDING_DIM = 1536  # text-embedding-ada-002


def get_qdrant_client():
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


def collection_has_docs(client: QdrantClient) -> bool:
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            return False
        info = client.get_collection(COLLECTION_NAME)
        return (info.points_count or 0) > 0
    except Exception:
        return False


def index_documents(client: QdrantClient):
    """Load the source file, embed chunks, and upsert into Qdrant."""
    if not os.path.exists(DOCS_PATH):
        st.error(f"Document file not found: {DOCS_PATH}")
        return False

    loader = TextLoader(DOCS_PATH)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    # Recreate the collection so we start clean
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    Qdrant.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )
    return True


def build_chain(client: QdrantClient):
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def main():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("PGX Chatbot")
    with col2:
        logo = os.getenv("LOGO_PATH", "")
        if logo and os.path.exists(logo):
            st.image(logo, width=150)

    # --- RAG status sidebar ---
    client = get_qdrant_client()
    with st.sidebar:
        st.subheader("RAG Source")
        if collection_has_docs(client):
            info = client.get_collection(COLLECTION_NAME)
            st.success(f"{info.points_count} chunks indexed  \ncollection: `{COLLECTION_NAME}`")
        else:
            st.warning("No docs indexed")
            if st.button("Index documents now"):
                with st.spinner("Indexing…"):
                    ok = index_documents(client)
                if ok:
                    st.success("Indexing complete — please refresh.")
                    st.experimental_rerun()

    if not collection_has_docs(client):
        st.info("No documents are indexed yet. Use the sidebar to index your file.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "model", "message": "Hi! I'm here to answer your questions about PGX"}
        ]

    qa_chain = build_chain(client)

    user_input = st.chat_input("Ask a question about PGX:")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        retrieved_docs = qa_chain.retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = (
            "You are an information assistant providing answers based on the documents provided. "
            "Here is the relevant information:\n"
            f"{context}\n\n"
            f"User asked: {user_input}\n"
            "Please provide the dosing recommendations in a table format with three columns: "
            "Phenotype, Drug, and Dosing Recommendations."
        )

        response = qa_chain.run(prompt)
        st.session_state.chat_history.append({"role": "model", "message": response})

    for chat in st.session_state.chat_history:
        role = "user" if chat["role"] == "user" else "assistant"
        st.chat_message(role).write(chat["message"])


if __name__ == "__main__":
    main()
