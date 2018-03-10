import helpers
# some of the deep processing takes a lot of time..
import datetime
# I hope you like some deep learning
print("Importing Machine Learning libraries.. ", end="")
# import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Embedding, Conv2D, SimpleRNN, Dropout
from keras.callbacks import History
print("Done.")

# Long Short-Term Memory (15 marks)

# The LSTM will use sequences of word2vec features extracted from each article. Set the maximum sequence length to 1000. Use zero-padding for shorter sentences (in Keras you can use the pad_sequencesutility function) In Keras you may use an Embedding input layer (https://keras.io/layers/embeddings/) to map the word2vec features into a structure that Keras can process. Remember to split your data into training and testing sets. When working with LSTMs start experimenting with a subset of the data until you are satisfied with your architecture and then run the model on all the training data. This will save you time when debugging your code or deciding on model parameters.

# -----------------------------
# Pre Processing and Feature Extraction
# -----------------------------


# Write a function that takes a text as input and returns a list of 
# word2vec features using a pre-trained word2vec model. (5 Marks)

# Spacy (https://spacy.io/) is an example of a package that provides pre-trained Word2Vec models for English.

def word2vec(dataset, filename="word2vec"):
  print("Initialising Word2Vec modelling.. ")
  # https://github.com/lesley2958/word2vec
  # http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
  # https://radimrehurek.com/gensim/models/word2vec.html
  # https://learn.adicu.com/word2vec/
  # http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
  # http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
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

def generateGloveDict(gloveWords, debug=False):
  if debug:
    print("Generating word2Vec dictionary.. ", end="")
  gloves = dict(zip(gloveWords, range(len(gloveWords))))
  if debug:
    print("Done.")
  return gloves

def word2int(dataset, gloveWords, glove, threshold=2, debug=True):
  if debug:
    print("Generating text2int.. ", end="")
  # generate a list of words in glove
  text2int = generateGloveDict(gloveWords)

  # consider words that have been used more than our threshold
  popularWords = []
  for word in dataset['words']:
    # check if their document frequency is greater than some threshold
    if dataset['words'][word]['df'] > threshold:
  # get ids of words that have been considered more than our threshold
      popularWords.append(word)

  # make a list of the popular words and see whether they're in the gloves
  genericWords = []
  absentWords = []
  for word in popularWords:
    if word in text2int.keys():
      genericWords.append(word)
    else:
      absentWords.append(word)


  # also add null values for words that don't meet the threshold
  null_value = "!NULL!"
  # also add padding value for articles that are too short
  padding_value = "!PAD!"

  # -------------------------------

  # Create a vector that is the average of the last 1000 words in the glove
  # dataset.
  lastFewMatrix = []
  for i in list(gloveWords)[-1000:]:
    lastFewMatrix.append(glove[i])
  lastFewMatrix = np.array(lastFewMatrix).mean(axis=0)

  # create a null entry that we'll use for null and padding values.
  # dudEntry = np.zeros(glove["and"].shape)
  dudEntry = lastFewMatrix
  # create a smaller glove dataset using only the words we'll need
  # to consider
  microGlove = np.zeros((len(genericWords)+2, glove["and"].shape[0]))
  print("Creating micro glove with shape", microGlove.shape)

  microGloveText2Ints = {
    null_value : 0,
    padding_value : 1
  }
  incrementer = 2
  # now we need to remove the words in the glove dataset
  # that isn't considered in the training data
  # otherwise we're just wasting memory
  for word in genericWords:
    microGlove[incrementer] = glove[word]
    microGloveText2Ints[word] = incrementer
    incrementer += 1
  for word in absentWords:
    microGloveText2Ints[word] = 0

  # now we can return the smaller word2vec!
  return (microGlove,microGloveText2Ints)

def word2Glove(article, glove, glovewords):
  """
  This is only used as a demonstration,
  The actual implementation that uses word2vec -> keras is used differently.
  I just want to get the marks ¯\_(ツ)_/¯
  """
  ArticleGlove = []
  # also need a null entry for words that aren't in the glove
  # dataset
  nullVector = np.zeros(glove[0].shape)
  # iterate through the words and make a vector of it!
  for word in article:
    # check if this word exists in the glove set
    if word in glovewords:
      ArticleGlove.append(glove[glovewords[word]])
    else:
      ArticleGlove.append(nullVector)
  # convert it into an numpy array and voila!
  return np.array(ArticleGlove)

def int2word(word2int):
  back = {}
  for word in word2int:
    back[word2int[word]] = word
  return back

def reverseText(word2intText,word2int):
  back = []
  rev = int2word(word2int)
  for i in word2intText:
    back.append(rev[i])
  print(back)

def text2int(article, word2int, wordLimit=1000, voidEntries=False):
  # converts an individual article's text to integers
  articleInInts = []
  # convert an array of words to an array of ints based on our word2int
  articleSize = len(article)

  # if the article is too long, cut it off at the wordlimit
  if articleSize > wordLimit:
    articleSize = wordLimit

  for i in range(articleSize):
    word = article[i]
    if word in word2int:
      articleInInts.append(word2int[word])
    else:
      if voidEntries:
        articleInInts.append(word2int["!NULL!"])
  # if the article is pretty short, we need to pad the rest of the article
  # so it meets the wordlimit size.
  remainder = wordLimit - len(articleInInts)
  for i in range(remainder):
    padding = []
    padding.append(word2int["!PAD!"])
    articleInInts = padding + articleInInts
  
  return articleInInts

def convertArticlesToInts(dataset, word2int, wordLimit=1000):
  # converts all articles into word2ints
  for article in dataset['data']:
    # compute the text2int for an article
    article_array = dataset['data'][article]['data']
    data = text2int(article_array, word2int, wordLimit)
    dataset['data'][article]['t2i'] = data
    # break
  return dataset


# -----------------------------
# Classification
# -----------------------------


def splitTrainingData(dataset):
  x_train, y_train, x_test, y_test = [], [], [], []

  for i in dataset['test_data']:
    if not dataset['test_data'][i]:
      x_train.append(dataset['data'][i]['t2i'])
      y_train.append(dataset['data'][i]['class'])
    else:
      x_test.append(dataset['data'][i]['t2i'])
      y_test.append(dataset['data'][i]['class'])

  # convert the list into np arrays
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  x_test = np.array(x_test)
  y_test = np.array(y_test)
  return {
    'train' : {
      'x' : x_train,
      'y' : y_train
    },
    'test' : {
      'x' : x_test,
      'y' : y_test
    }
  }

def makeNN(weights, netType="lstm", useWeights=False, activation="sigmoid", printSummary=False, useDropout=False):
  print("Initialising Neural Net ("+netType+")... ", end="")
  # we'll use a sequential RNN
  model = keras.models.Sequential()

  # initialise the embedding layer
  if useWeights:
    model.add(Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=False))
  else:
    model.add(Embedding(weights.shape[0], weights.shape[1]))

  if netType == "rnn":
    model.add(SimpleRNN(64)) # initialise a recurrent layer
  else:
    model.add(LSTM(64)) # initialise the LSTM layer size

  # add a dropout layer
  if useDropout:
    model.add(Dropout(0.1))

  # add a dense layer
  model.add(Dense(1, activation=activation))
  # group them together!
  model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

  if printSummary:
    # print the summary
    model.summary()
  print("Done.")
  return model

def runNN(model, trainingData, history, epochs=2, netType="rnn"):
  print("Running Neural Net (" + netType+")..")
  xtr = trainingData['train']['x']
  ytr = trainingData['train']['y']
  xte = trainingData['test']['x']
  yte = trainingData['test']['y']
  # how many samples to use
  batch_size = 128
  # epochs is the number of passes over the dataset
  # start training!
  model.fit(xtr, ytr, batch_size=batch_size, epochs=epochs, validation_split=0.05,  callbacks=[history], verbose=2)
  s = model.predict(xte, batch_size=batch_size)

  # convert it in a variable that can be interpreted with
  # our evaluator
  results = []
  for i in range(len(s)):
    results.append({
      'guess' : np.rint(s[i]),
      'actual' : yte[i]
    })
  
  print("Finished running neural net(" + netType+").")
  return (model, results)

def getMeanMeasurements(histories):
  averaged = {}
  # get the measurement keys and add them to our avg
  for i in histories[0].keys():
    # print(i, histories[0][i])
    averaged[i] = []
  # get each runthrough and add it to our list
  for i in histories:
    for measurement in i.keys():
      averaged[measurement].append(np.mean(i[measurement]))
  # calculate the mean of the results
  for i in averaged.keys():
    averaged[i] = np.round(np.mean(averaged[i]),3)
  # return the results
  return averaged

def evaluateActivationFunctions(data, glove, epochs=1, loops=10):
  """
  This function is used to determine the effectiveness of different
  activation functions we have at our disposal. The report discusses
  using tanh as it is the best overall contender. However,
  the tests itself can be rerun.
  """
  # activationMethods = ["sigmoid", "relu", "tanh", "linear"]
  activationMethods = ["sigmoid"]
  mass_history = {}
  print("--")
  for activation in activationMethods:
    # store those deep results!
    overallLSTMHistory = []
    overallRNNHistory = []
    for i in range(loops):
      # set up neural networks (thanks keras)
      lstm_model = makeNN(glove,"lstm",activation=activation,useWeights=True)
      rnn_model = makeNN(glove,"rnn",activation=activation,useWeights=True)
      lstm_h = History()
      rnn_h = History()
      # run models on keras
      rnn_results = runNN(rnn_model, data, rnn_h, epochs, "rnn")
      lstm_results = runNN(lstm_model, data, lstm_h, epochs, "lstm")
      overallLSTMHistory.append(evaluate(lstm_h))
      overallRNNHistory.append(evaluate(rnn_h))
    # now we average the results and make an averaged list.
    mass_history[activation] = {
      "rnn " : getMeanMeasurements(overallRNNHistory),
      "lstm" : getMeanMeasurements(overallLSTMHistory)
    }

  # print the list for future analysis
  for i in mass_history:
    print(i)
    print(mass_history[i])
  print()

if __name__ == "__main__":
  print("Testing LSTM")
  glove = helpers.loadGlove()
  gloveWords = glove.keys()
  import shallow
  import helpers
  # load our dataset.
  dataset = helpers.loadJSON()
  dataset = shallow.tf(dataset)
  dataset = shallow.df(dataset)

  # convert words to integers using our standardised glove dataset
  # this is so we can process them in our neural networks
  (glove,word2int) = word2int(dataset,gloveWords,glove,threshold=2)
  dataset = convertArticlesToInts(dataset, word2int, wordLimit=1000)
  # split data
  data = splitTrainingData(dataset)

  # evaluate the different activation functions
  # evaluateActivationFunctions(data, glove, epochs=10)
  
  # set number of rounds
  epochs = 1

  # # set up neural network models
  lstm_model = makeNN(glove,"lstm",activation="sigmoid",useWeights=True,useDropout=False)
  rnn_model = makeNN(glove,"rnn",activation="sigmoid",useWeights=True,useDropout=False)
  lstm_h = History()
  rnn_h = History()
  # run models on keras
  (_, rnn_results) = runNN(rnn_model, data, rnn_h, epochs, "rnn")
  (_, lstm_results) = runNN(lstm_model, data, lstm_h, epochs, "lstm")
  print("LSTM (With Weights):",helpers.evaluate(lstm_results))
  print("RNN (With Weights): ",helpers.evaluate(rnn_results))

  print("Done")
