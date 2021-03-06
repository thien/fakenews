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
from keras.layers import Activation, Dense, LSTM, Embedding, Conv2D, SimpleRNN
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

# -----------------------------
# Classification
# -----------------------------


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

  # create a null entry that we'll use for null and padding values.
  dudEntry = np.zeros(glove["and"].shape)
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
  # iterate through the words and make a vector o fit!
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

def text2int(article, word2int, wordLimit=1000):
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
      articleInInts.append(word2int["!NULL!"])
  # if the article is pretty short, we need to pad the rest of the article
  # so it meets the wordlimit size.
  remainder = wordLimit - articleSize
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

# Precision, measure and f1 measures were taken away from keras on 18th jan 2017. Fortunately, this can be reused again by looking at the keras commits on github.
# https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7#diff-7b49e1c42728a58a9d08643a79f44cd4

def precision(y_true, y_pred):
  """
  Precision metric.
  Only computes a batch-wise average of precision.
  Computes the precision, a metric for multi-label classification of
  how many selected items are relevant.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def recall(y_true, y_pred):
  """
  Recall metric.
  Only computes a batch-wise average of recall.
  Computes the recall, a metric for multi-label classification of
  how many relevant items are selected.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def fbeta_score(y_true, y_pred, beta=1):
  """Computes the F score.

  The F score is the weighted harmonic mean of precision and recall.
  Here it is only computed as a batchwise average, not globally.

  This is useful for multilabel classification, where input samples can be
  classified as sets of labels. By only using accuracy (precision) a model
  would achieve a perfect score by simply assigning every class to every
  input. In order to avoid this, a metric should penalize incorrect class
  assignments as well (recall). The Fbeta score (ranged from 0.0 to 1.0)
  computes this, as a weighted mean of the proportion of correct class
  assignments vs. the proportion of incorrect class assignments.

  With beta = 1, this is equivalent to a Fmeasure. With beta < 1, assigning
  correct classes becomes more important, and with beta > 1 the metric is
  instead weighted towards penalizing incorrect class assignments.
  """
  if beta < 0:
      raise ValueError('The lowest choosable beta is zero (only precision).')

  # If there are no true positives, fix the F score at 0 like sklearn.
  if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
      return 0

  p = precision(y_true, y_pred)
  r = recall(y_true, y_pred)
  bb = beta ** 2
  fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
  return fbeta_score

def fmeasure(y_true, y_pred):
  """Computes the fmeasure, the harmonic mean of precision and recall.

  Here it is only computed as a batchwise average, not globally.
  """
  return fbeta_score(y_true, y_pred, beta=1)

def makeNN(weights, netType="lstm", useWeights=False, activation="tanh", printSummary=False):
  print("Initialising Neural Net ("+netType+")... ", end="")
  # we'll use a sequential CNN
  model = keras.models.Sequential()

  # initialise the embedding layer
  if useWeights:
    model.add(Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=False))
  else:
    model.add(Embedding(weights.shape[0], weights.shape[1]))

  if netType == "cnn":
    model.add(SimpleRNN(64)) # initialise a recurrent layer
  else:
    model.add(LSTM(64)) # initialise the LSTM layer size
    
  # initialise the dense layer size (this is the actual hidden layer)
  model.add(Dense(1, activation=activation))
  # group them together!
  model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy', precision, recall, fmeasure])

  if printSummary:
    # print the summary
    model.summary()
  print("Done.")
  return model

def runNN(model, trainingData, history, epochs=2, netType="cnn"):
  print("Running Neural Net (" + netType+")..")
  xtr = trainingData['train']['x']
  ytr = trainingData['train']['y']
  xte = trainingData['test']['x']
  yte = trainingData['test']['y']
  # how many samples to use
  batch_size = 128
  # epochs is the number of passes over the dataset
  # start training!
  model.fit(xtr, ytr, batch_size=batch_size, epochs=epochs, validation_split=0.0, validation_data=(xte, yte),  callbacks=[history], verbose=2)
  print("Finished running neural net(" + netType+").")
  return model

def evaluate(hist):
  # val_loss is the value of cost function for your cross validation data and loss is the value of cost function for your training data. On validation data, neurons using drop out do not drop random neurons. The reason is that during training we use drop out in order to add some noise for avoiding over-fitting. During calculating cross validation, we are in recall phase and not in training phase. We use all the capabilities of the network.

  # we want to keep the results that consider (cross) validation data, which are keys that include val_. therefore, we remove keys that don't include this.

  # removing the scores of the training data since that's not what we're measuring.
  bad = []
  for i in hist.history:
    if "val_" not in i:
      bad.append(i)
  for i in bad:
    hist.history.pop(i, None)
  
  # rename the remaining keys by removing the "val_" section
  newHist = {}
  for i in hist.history:
    new_i = i.replace("val_", "")
    newHist[new_i] = hist.history[i]
  hist.history = newHist

  # round results to 3dp and on a scale of 0-100%
  for i in hist.history:
    hist.history[i] = np.round(np.multiply(hist.history[i],100),3).tolist()

  return hist.history

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
  activationMethods = ["sigmoid", "relu", "tanh", "linear"]
  mass_history = {}
  print("--")
  for activation in activationMethods:
    # store those deep results!
    overallLSTMHistory = []
    overallCNNHistory = []
    for i in range(loops):
      # set up neural networks (thanks keras)
      lstm_model = makeNN(glove,"lstm",activation)
      cnn_model = makeNN(glove,"cnn",activation)
      lstm_h = History()
      cnn_h = History()
      # run models on keras
      cnn_results = runNN(cnn_model, data, cnn_h, epochs, "cnn")
      lstm_results = runNN(lstm_model, data, lstm_h, epochs, "lstm")
      overallLSTMHistory.append(evaluate(lstm_h))
      overallCNNHistory.append(evaluate(cnn_h))
    # now we average the results and make an averaged list.
    mass_history[activation] = {
      "cnn " : getMeanMeasurements(overallCNNHistory),
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
  # evaluateActivationFunctions(data, glove)
  
  # set number of rounds
  epochs = 1

  # set up neural network models
  lstm_model = makeNN(glove,"lstm",activation="sigmoid")
  cnn_model = makeNN(glove,"cnn",activation="sigmoid")
  lstm_h = History()
  cnn_h = History()
  # run models on keras
  cnn_results = runNN(cnn_model, data, cnn_h, epochs, "cnn")
  lstm_results = runNN(lstm_model, data, lstm_h, epochs, "lstm")
  print("LSTM:",evaluate(lstm_h))
  print("CNN: ",evaluate(cnn_h))

  print("Done")
