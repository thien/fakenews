# I hope you like some deep learning
import spacy
from gensim.models import Word2Vec
# -----------------------------
# Pre Processing and Feature Extraction
# -----------------------------


# Write a function that takes a text as input and returns a list of 
# word2vec features using a pre-trained word2vec model. (5 Marks)

# Spacy (https://spacy.io/) is an example of a package that provides pre-trained Word2Vec models for English.

def word2vec(dataset, filename="word2vec"):
  print("Initialising Word2Vec modelling.. ")
  # http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
  # https://radimrehurek.com/gensim/models/word2vec.html
  model = None
  try:
    model = Word2Vec.load(filename)
    print(filename, "loaded into models.")
  except:
    print("WARNING: a model isn't saved, so one will be generated.")
    print("Loading Dataset.. ", end='')
    sentences = [dataset['data'][x]['data'] for x in dataset['data']]
    print("Done.\nRunning Word2Vec generator.. ", end="")

    # I am google, spacy and Fast-Text bruh
    hiddenLayerSize = 100
    # Window size is the size of the neighborhood
    windowSize = 5

    model = Word2Vec(sentences, size=hiddenLayerSize, window=100, min_count=5, workers=4)
    print("Done.")

    print("Saving model to file.. ")
    model.save(filename)
    print("saved to", filename)
  
  trumping_results = model.most_similar('trump', topn=20)
  print("Most relevant keywords on 'Trump':")
  for i in trumping_results:
    print(i)
  return model

# -----------------------------
# Classification
# -----------------------------

# Long Short-Term Memory (15 marks)

# The LSTM will use sequences of word2vec features extracted from each article. Set the maximum sequence length to 1000. Use zero-padding for shorter sentences (in Keras you can use the pad_sequencesutility function) In Keras you may use an Embedding input layer (https://keras.io/layers/embeddings/) to map the word2vec features into a structure that Keras can process. Remember to split your data into training and testing sets. When working with LSTMs start experimenting with a subset of the data until you are satisfied with your architecture and then run the model on all the training data. This will save you time when debugging your code or deciding on model parameters.
  
def lstm(dataset):
  return 0

# Recursive Neural Network (10 marks)
def rnn(dataset):
  return 0

