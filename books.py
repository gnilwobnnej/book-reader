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

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, scrolledtext
from tkinter import ttk
import threading



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

    

'''main class
called when the app starts and sets up the GUI layout and widgets.
'''
class PDFQAAPP:
    def __init__(self, root):
        self.root= root
        self.root.title("Q&A with a Llama")
        self.qa_chain = None
        self.conversation = []

        self.setup_ui()

# sets up the ui and menu with the export conversation, pdf load button
# progress bar question entry and the ask button and scollable output text box

    def setup_ui(self):
        #menu bar
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        file_menu = tk.Menu(self.menu, tearoff=0)
        file_menu.add_command(label="Load PDF", command=self.load_pdf)
        file_menu.add_command(label="Export Convo", command = self.export_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu.add_cascade(label= "File", menu=file_menu)

        #main content frame
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.output_area = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD, height=20, width= 100)
        self.output_area.pack(fill="both", expand=True, pady=(0,10))
        
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress.pack(fill="x", side="top", padx=10, pady=(0,10))

        #bottom frame from question input and ask button
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(fill="x", side="bottom", padx=10, pady=(0,10))


        self.placeholder_text = "Type your question here."
        self.question_entry = tk.Entry(self.bottom_frame, width=80, fg="grey")
        self.question_entry.insert(0, self.placeholder_text)
        self.question_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.ask_button = tk.Button(self.bottom_frame, text="Ask", command=self.ask_question)
        self.ask_button.pack(side="right")


        #remove placeholder on focus, restore placeholder if it's empty binder enter to ask
        self.question_entry.bind("<FocusIn>", self.clear_placeholder)
        self.question_entry.bind("<FocusOut>", self.add_placeholder)
        self.question_entry.bind("<Return>", lambda event: self.ask_question())

    def clear_placeholder(self, event):
        if self.question_entry.get() == self.placeholder_text:
            self.question_entry.delete(0, tk.END)
            self.question_entry.config(fg="black")

    def add_placeholder(self, event):
        if not self.question_entry.get():
            self.question_entry.insert(0, self.placeholder_text)
            self.question_entry.config(fg="grey")



#lets user pick the pdf file starts the process of processing the pdf
    def load_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return
        

        self.output_area.insert(tk.END, "reading PDF...\n")
        self.progress.start()
        threading.Thread(target=self.process_pdf, args=(file_path,)).start()

#runs in the background, reads the pdf, splits it into chunks, 
#builds vector store and QA chain stops the loading animation
    def process_pdf(self, file_path):
        try:
            text = read_pdf(file_path)
            chunks = split_text(text)
            self.output_area.insert(tk.END, "Creating vectorstore...\n")
            vectorstore = create_vectorstore(chunks)
            self.output_area.insert(tk.END, "Building QA system...\n")
            self.qa_chain = create_qa_chain(vectorstore)
            self.output_area.insert(tk.END, "Ready! Ask your quetions.\n\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally: 
            self.progress.stop()

##gets the questions from the text box, sends it to the qa system, displays the answer in the output area,
# logs the interaction to a file using threading so it doesn't freeze.
    def ask_question(self):
        if not self.qa_chain:
            messagebox.showwarning("Warning", "Please load PDF first.")
            return
        
        query = self.question_entry.get().strip()
        if not query:
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.output_area.insert(tk.END, f"[{timestamp}]You: {query}\n")
        self.conversation.append(f"[{timestamp}]\nYou: {query}")
        self.question_entry.delete(0, tk.END)
        self.progress.start()

        def run_query():
            try:
                answer = self.qa_chain.invoke({"query": query})["result"]
                wrapped = "\n".join(textwrap.fill(p, WRAP) for p in answer.split('\n'))

                def update_ui():
                    response_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.output_area.insert(tk.END, f"[{response_time}]Llama: \n{wrapped}\n\n")
                    self.conversation.append(f"[{response_time}]Llama\n{wrapped}{'+' * WRAP}")
                    self.progress.stop()

                self.root.after(0, update_ui)
            except Exception as e:
                self.root.after(0, lambda: self.output_area.insert(tk.END, f"Error: {e}\n"))
                self.root.after(0, self.progress.stop)


        threading.Thread(target=run_query).start()

#lets the user to save the conversation to a text file.
    def export_conversation(self):
        export_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text File", "*.txt")])
        if not export_path:
            return
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(self.conversation))
            messagebox.showinfo("Export", f"Conversation exported to:\n {export_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")



'''Just runs the main function'''
if __name__ == "__main__":
    root = tk.Tk()
    app = PDFQAAPP(root)
    root.mainloop()