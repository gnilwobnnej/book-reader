#fitz brings in the pdf
import fitz 
#splits large blocks of txt to smaller chunks
from langchain.text_splitter import CharacterTextSplitter 
#helps build the questioning and answering
from langchain.chains import RetrievalQA
#smart search engine
from langchain_community.vectorstores import FAISS
#Embeddings turn text into numbers so the computer can understand and compare meaning
#uses local language model to generate a response
from langchain_ollama import OllamaEmbeddings, OllamaLLM
#wraps the text
import textwrap
import datetime
import os


MODEL = "gemma2"
WRAP = 80


'''reads the book 'book.pdf'
#loops through every page and extracts the text/combines all the text to one string
#returs the txt '''
def read_pdf(file_path):
    doc = fitz.open(file_path)
    return "".join(page.get_text() for page in doc)

'''splits to chunks
chunk_size=1000: splits the text into pieces 1000 characters long.
chunk_overlap=100: overlaps 100 characters between chunks so the meaning isn’t lost between splits.'''
def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

'''creates vector store
uses the gemma model to turn each chunk into a vector (a list of numbers representing meaning)
stores those vectors in FAISS, which is a fast search database for vectors'''
def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=MODEL)
    return FAISS.from_texts(chunks, embedding=embeddings)

'''creates the questions and answer with ollama
uses a constant that can be updated above.
vectorstore.as_retriever turns the stored chunks into something ai can search k=x is when a
question is asked it will find the x most relevant chunks more being more indepth it will be
'''
def create_qa_chain(vectorstore):
    llm = OllamaLLM(model=MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#get the file
def get_log_filename():
    while True:
        name = input("Enter a file name to save the conversation as: ").strip()
        if not name:
            print("Filename cannot be empty. Please try again.")
            continue
        if not name.lower().endswith(".txt"):
            name +=".txt"
        if os.path.exists(name):
            confirm = input(f"'{name}' already exists. Overwrite? (y/n)").strip().lower()
            if confirm != 'y':
                continue
        return name
    
#upload pdf
def get_pdf_filename():
    while True:
        file_path= input("Enter the PDF file to read: ").strip()
        if not file_path:
            print("PDF filename cannot be empty. Try again.")
            continue
        if not file_path.lower().endswith(".pdf"):
            file_path += ".pdf"
        if not os.path.exists(file_path):
            print(f"'{file_path}' not found. Try again.")
            continue
        return file_path
    
#log interations uses timestamps to log each individual query
def log_interaction(log_file, question, answer):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"You: {textwrap.fill(question, width=WRAP)}\n")
        f.write("Llama: \n")
        for paragraph in answer.split('\n'):
            f.write(textwrap.fill(paragraph, width=WRAP) + "\n")
        f.write("=" * WRAP + "\n")


'''looplooploop
main process- read > split > creates qa system
starts a loop where you can type questions, 'exit' will quit.
will loop question and then answers.
save loop with a textfile that is wrapped.
'''
def main():

    log_file= get_log_filename()
    pdf_file= get_pdf_filename()

    print("Reading PDF...")
    text = read_pdf(pdf_file)

    print("Splitting text...")
    chunks = split_text(text)

    print("Creating vectorstore...")
    vectorstore = create_vectorstore(chunks)

    print("Building QA chain...")
    qa_chain = create_qa_chain(vectorstore)

#the loop and adds the textwrap with 80 characters.
    print("\nAsk your PDF questions (type 'exit' to quit):")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print(f"\nConversation saved to {log_file}")
            break
        answer = qa_chain.invoke({"query": query})
        wrapped_answer = "\n".join(textwrap.fill(p, width=WRAP) for p in answer['result'].split('\n'))
        print(wrapped_answer + "\n")

        log_interaction(log_file, query, answer['result'])

'''Just runs the main function'''
if __name__ == "__main__":
    main()