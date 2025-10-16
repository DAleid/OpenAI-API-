🧠 Description of the PGX Chatbot Code

This Streamlit-based Python application implements a Retrieval-Augmented Generation (RAG) chatbot designed to assist healthcare professionals with pharmacogenomics (PGX)-related questions,
specifically focusing on TPMT genotype-based dosing recommendations.

🔍 How It Works

Document Loading
The app loads a text document (TPMT_doc_cleaned.txt) that contains pharmacogenomic guideline information. The text is split into smaller,
overlapping chunks using the CharacterTextSplitter from Langchain to improve retrieval accuracy.

Embedding and Index Creation
Each text chunk is converted into numerical vector embeddings using OpenAI’s Embedding API (OpenAIEmbeddings).
These embeddings are stored in a FAISS vector database, which allows efficient semantic search for relevant text passages when a question is asked.

Retriever and LLM Setup
The FAISS retriever is connected to a GPT-3.5-turbo model via Langchain’s RetrievalQA chain. 
This setup allows the model to retrieve the most relevant context from the documents before generating a response — a key part of the RAG approach.

Streamlit Chat Interface
The chatbot interface is built with Streamlit:

Displays the title “PGX Chatbot” along with the King Abdulaziz City for Science and Technology (KACST) logo.

Accepts user questions in natural language.

Displays the full chat history dynamically (both user and model messages).

Question Answering Process
When a user enters a query:

The retriever fetches the most relevant document sections.

A context-rich prompt is built that includes the retrieved information.

The GPT-3.5 model then generates a structured answer, typically showing Phenotype, Drug, and Dosing Recommendations in a table format.

The chat history is updated and displayed to maintain a continuous conversation flow.

⚙️ Technologies Used

Streamlit – for interactive user interface

Langchain – to manage the retrieval and LLM pipeline

OpenAI GPT-3.5-turbo – for text generation

FAISS – for vector-based semantic search

Python – for overall application logic

🎯 Purpose

The chatbot aims to enhance clinical decision support by providing quick, evidence-based dosing recommendations derived from pharmacogenomic guidelines. It helps bridge the gap between complex genetic data and actionable treatment guidance for healthcare professionals.
