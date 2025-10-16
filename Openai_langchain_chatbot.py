import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter

# Set your OpenAI API key

api_key = os.getenv("OPENAI_API_KEY")
def load_documents(path):
    loader = TextLoader(path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_index(documents):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def initialize_chain():
    documents = load_documents("C:/Users/danyh/Downloads/TPMT_doc_cleaned.txt")
    vector_store = create_index(documents)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

# Function to truncate chat history to fit within token limit


def main():
   # Create two columns: one for the title and one for the image
    col1, col2 = st.columns([4, 1])  # Adjust the column width proportions as needed

    with col1:
        st.title("PGX Chatbot")

    with col2:
        st.image("C:/Users/danyh/Desktop/coop/شعار-مدينة-الملك-عبدالعزيز-للعلوم-والتقنية.png", width=150)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "model", "message": "Hi! I'm here to answer your questions about PGX"}]

    qa_chain = initialize_chain()

    user_input = st.chat_input("Ask a question about PGX:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        ##st.session_state.chat_history = truncate_chat_history(st.session_state.chat_history)

        # Retrieve relevant context based on user input
        retrieved_docs = qa_chain.retriever.get_relevant_documents(user_input)
        
        # Combine the retrieved documents into a context string
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        history_context = "\n".join([f'{chat["role"]}: {chat["message"]}' for chat in st.session_state.chat_history])

        # Construct a prompt for the LLM using the retrieved context
        prompt = (
            "You are an information assistant providing answers based on the documents provided. "
            "Here is the relevant information:\n"
            f"{context}\n\n"
            
            f"User asked: {user_input}\n"
            "Please provide the dosing recommendations in a table format with three columns: "
            "Phenotype, Drug, and Dosing Recommendations."
            
        )

        # Get the response from the LLM
        response = qa_chain.run(prompt)
        st.session_state.chat_history.append({"role": "model", "message": response})

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.chat_message("user").write(chat["message"])
        else:
            st.chat_message("assistant").write(chat["message"])

if __name__ == "__main__":
    main()
