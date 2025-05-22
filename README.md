# book-reader
This program uses [Ollama](https://ollama.com/) <img src="https://ollama.com/public/ollama.png" alt="ollama" height="75"> using the [Mistral](https://ollama.com/library/mistral-small3.1) model <img src= "https://ollama.com/assets/library/mistral-small3.1/88f81c26-7028-4f08-b906-92b873d5536e" alt="mistral" height="50"> to run the LLM locally. 
<br>
<br>
bookgemma.py uses [Gemma2](https://ollama.com/library/gemma2) <img src="https://ollama.com/assets/library/gemma2/58a4be20-b402-4dfa-8f1d-05d820f1204f" alt="gemma2" width="200">
<br>
<br>
In the example screenshots I used the book [Animal Farm by George Orwell](https://en.wikipedia.org/wiki/Animal_Farm)  :pig: (not included in the upload)
<br>
<br>
Can be used to read other pdf's, it's looking for a file labled 'book.pdf', that can be changed on line 56 in book.py 57 in bookgemma.py: 
```python
    text = read_pdf("book.pdf")
```

## UPDATE 5/16/2025

1. Made the models constants so I can easliy go back and change them if I want to try other ones, this one is for bookgemma.py
2. Made it so that you can choose the name of the pdf that you want.
3. Made so you save the file as you want it and so that the text inputs and outputs are saved.
4. 'this is a test.txt' is the test file that is used using gemma2 and the new format.
5. Will need to try with other models. 


## UPDATE 5/22/2025

1. made and updated a separate program that launches a gui 
2. made it so that the user can upload any pdf with ease
3. have it so that users can save the text file of the conversation.

## TODO LIST
-[] fix it so that it doesn't save the chat first off and export save a separate chat
-[] update the gui so it looks better
-[] try a different book