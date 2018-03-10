# KaggleDathena


# files
## Notebooks

`DataFrameCreation.ipynb` : Read and exctract freatures from all documents and save it as a pandas dataFrame (in `safe/df_file_final.csv`) 

`Pipline_Vx.x.ipynb`: Main notebook that create our pipline for this project (using the other modules) and that create the submission file

`Test.ipynb`: Some unit test for utility functions

`Train_sentiment_analyser.ipynb`: Notebook to exectue to train our sentiment annalyser CNN. Warring you better have a good **GPU** to train it.
The training is done either on [Amazon reviews](https://www.kaggle.com/bittlingmayer/amazonreviews/data) or imdb movie review using the [keras dataset](https://keras.io/datasets/).

`Train_word_embeddings.ipynb`: based on the Corpus (all document) this learn a vector representation for each lemma (word) and a [mapping dictionary](safe/dico.p).


## Python code (module)
`extract_text.py`: some function to extract text data from **doc(x)**, **pdf** and **xls(x)** files

`name_entity_analyzer.py`: code to detect if a document is about the company (**Keppel** or **Prudential**)

`text_preprocessing.py`: code to preprocess the text, like doing some Lemmatisation, vectorization, stop words removal as well as some regex cleaning.

`text_summarization.py`: code to extract interesting sentence about the companies.

`util.py`: Some utility function which doesn't find a place in other an file.

## Others
`submission_mapper.csv`: provided file slightly modified (*name of the files containing parenthesis*)


# Folders
`safe`: Contains checkpoint for faster exection of the code as pickle or csv for pandas

# Dependencies
```
pip install PyPDF2
pip install python-docx
pip install xlrd
pip install pdfrw
pip install sumy
pip install gensim
pip install nltk


python -m spacy download en
import nltk
nltk.download('punkt')
```