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



'''reads the book 'book.pdf'
#loops through every page and extracts the text/combines all the text to one string
#returs the txt '''
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

'''splits to chunks
chunk_size=1000: splits the text into pieces 1000 characters long.
chunk_overlap=100: overlaps 100 characters between chunks so the meaning isn’t lost between splits.'''
def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

'''creates vector store
uses the Mistral model to turn each chunk into a vector (a list of numbers representing meaning)
stores those vectors in FAISS, which is a fast search database for vectors'''
def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="mistral")
    return FAISS.from_texts(chunks, embedding=embeddings)

'''creates the questions and answer with ollama
uses mistral as the llm
vectorstore.as_retriever turns the stored chunks into something ai can search k=x is when a
question is asked it will find the x most relevant chunks more being more indepth it will be
'''
def create_qa_chain(vectorstore):
    llm = OllamaLLM(model="mistral")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

'''looplooploop
main process- read > split > creates qa system
starts a loop where you can type questions, 'exit' will quit.
will loop question and then answers.
'''
def main():
    print("Reading PDF...")
    text = read_pdf("book.pdf")
    print("Splitting text...")
    chunks = split_text(text)

    print("Creating vectorstore...")
    vectorstore = create_vectorstore(chunks)

    print("Building QA chain...")
    qa_chain = create_qa_chain(vectorstore)

    print("\nAsk your PDF questions (type 'exit' to quit):")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        answer = qa_chain.invoke({"query": query})
        print(f"Answer: {answer}\n")

'''Just runs the main function'''
if __name__ == "__main__":
    main()
