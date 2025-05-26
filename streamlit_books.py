import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import textwrap
import datetime
import tempfile
import os

MODEL = "gemma2"
WRAP = 80

def read_pdf(file_path):
    doc = fitz.open(file_path)
    return "".join(page.get_text() for page in doc)

def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=MODEL)
    return FAISS.from_texts(chunks, embedding=embeddings)

def create_qa_chain(vectorstore):
    llm = OllamaLLM(model=MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

st.set_page_config(page_title="PDF Q&A with Llama", layout="wide")
st.title("Q&A with a Llama (Ollama-powered)")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.conversation = []

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        text = read_pdf(tmp_path)
        chunks = split_text(text)

        st.write("Text extracted and split into chunks.")
        
        with st.spinner("Creating vectorstore..."):
            vectorstore = create_vectorstore(chunks)

        with st.spinner("Building QA system..."):
            st.session_state.qa_chain = create_qa_chain(vectorstore)

        st.success("System ready! Ask your questions below.")

# Ask a question
if st.session_state.qa_chain:
    query = st.text_input("Ask a question:")

    if query:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.conversation.append(f"[{timestamp}] You: {query}")

        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": query})
                answer = result["result"]
                wrapped = "\n".join(textwrap.fill(p, WRAP) for p in answer.split('\n'))
                response_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.conversation.append(f"[{response_time}] Llama:\n{wrapped}")
            except Exception as e:
                st.error(f"Error during Q&A: {e}")

# Show conversation
if st.session_state.conversation:
    st.subheader("Conversation History")
    for entry in st.session_state.conversation:
        st.text(entry)

# Export conversation
if st.session_state.conversation:
    if st.download_button("Download Conversation", 
                          data="\n\n".join(st.session_state.conversation), 
                          file_name="conversation.txt"):
        st.success("Conversation downloaded!")