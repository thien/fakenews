# I hope you like some deep learning
import spacy
from gensim.models import Word2Vec
# -----------------------------
# Pre Processing and Feature Extraction
# -----------------------------


# Write a function that takes a text as input and returns a list of 
# word2vec features using a pre-trained word2vec model. (5 Marks)

# Spacy (https://spacy.io/) is an example of a package that provides pre-trained Word2Vec models for English.

def word2vec(dataset):
  print("Initialising Word2Vec modelling.. ")

  print("Loading Dataset.. ", end='')
  sentences = [dataset['data'][x]['data'] for x in dataset['data']]
  print("Done.\nRunning Word2Vec generator.. ", end="")

  # I am google, spacy and Fast-Text bruh
  hiddenLayerSize = 100
  # Window size is the size of the neighborhood
  windowSize = 5

  model = Word2Vec(sentences, size=hiddenLayerSize, window=10, min_count=5)
  print("Done.")

  return model

# -----------------------------
# Classification
# -----------------------------

# Long Short-Term Memory (15 marks)
def lstm(dataset):
  return 0

# Recursive Neural Network (10 marks)
def rnn(dataset):
  return 0

