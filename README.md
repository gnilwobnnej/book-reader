# book-reader
This program uses [Ollama](https://ollama.com/) <img src="https://ollama.com/public/ollama.png" alt="ollama" height="75"> using the [Mistral](https://ollama.com/library/mistral-small3.1) model <img src= "https://ollama.com/assets/library/mistral-small3.1/88f81c26-7028-4f08-b906-92b873d5536e" alt="mistral" height="50"> to run the LLM locally. 

In the example screenshots I used the book [Animal Farm by George Orwell](https://en.wikipedia.org/wiki/Animal_Farm)  :pig: (not included in the upload)

Can be used to read other pdf's, it's looking for a file labled 'book.pdf', that can be changed on line 56: 
```python
    text = read_pdf("book.pdf")
```

