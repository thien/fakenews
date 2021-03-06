import helpers

print("Importing Machine Learning libraries.. ", end="")
# import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Embedding, Conv2D, SimpleRNN
print("Done.")

def generateGloveDict(gloveWords, debug=False):
  if debug:
    print("Generating word2Vec dictionary.. ", end="")
  gloves = dict(zip(gloveWords, range(len(gloveWords))))
  if debug:
    print("Done.")
  return gloves

def word2int(dataset, gloveWords, threshold=2, debug=True):
  if debug:
    print("Generating text2int.. ", end="")
  # generate a list of words in glove
  text2int = generateGloveDict(gloveWords)
  sizeMatters = len(text2int)

  # consider words that have been used more than our threshold
  popularWords = []
  for word in dataset['words']:
    # check if their document frequency is greater than some threshold
    if dataset['words'][word]['df'] > threshold:
  # get ids of words that have been considered more than our threshold
      popularWords.append(word)

  # make a list of the popular words and see whether they're in the gloves
  genericWords = []
  for word in popularWords:
    if word in text2int.keys():
      genericWords.append(1)
    else:
      genericWords.append(0)
  
  # find words not in glove
  absent_words = []
  for word_ID in range(len(popularWords)):
    if genericWords[word_ID] == 0:
      absent_words.append(popularWords[word_ID])

  # create entries in our text2int for the absent words (which are also popular!)
  for word in absent_words:
    text2int[word] = sizeMatters
    sizeMatters += 1
  if debug:
    print("Done.")

  # also add null values for words that don't meet the threshold
  null_value = "!NULL!"
  text2int[null_value] = sizeMatters
  # also add padding value for articles that are too short
  padding_value = "!PAD!"
  text2int[padding_value] = int(sizeMatters)+1

  return text2int

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

def makeNN(word2num, netType="lstm", printSummary=False):
  print("Initialising Neural Net ("+netType+")... ", end="")
  # we'll use a sequential CNN
  model = keras.models.Sequential()
  # initialise the embedding layer size
  model.add(Embedding(len(word2num), 56))

  if netType == "cnn":
    # initialise a recurrent layer
    model.add(SimpleRNN(64))
  else:
    # initialise the LSTM layer size
    model.add(LSTM(64))

  # initialise the dense layer size (this is the actual hidden layer)
  model.add(Dense(1, activation='sigmoid'))
  # group them together!
  model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  if printSummary:
    # print the summary
    model.summary()
  print("Done.")
  return model

def runLSTM(model, trainingData, epochs=2):
  xtr = trainingData['train']['x']
  ytr = trainingData['train']['y']
  xte = trainingData['test']['x']
  yte = trainingData['test']['y']
  # how many samples to use
  batch_size = 128
  # epochs is the number of passes over the dataset
  # start training!
  model.fit(xtr, ytr, batch_size=batch_size, epochs=epochs, validation_data=(xte, yte))
  return model

if __name__ == "__main__":
  print("Testing LSTM")
  glove = helpers.loadGlove()
  gloveWords = glove.keys()
  import shallow
  # load our dataset.
  dataset = helpers.loadJSON()
  dataset = shallow.tf(dataset)
  dataset = shallow.df(dataset)

  word2int = word2int(dataset,gloveWords)
  dataset = convertArticlesToInts(dataset, word2int, wordLimit=1000)
  trainingData = splitTrainingData(dataset)
  lstm_model = makeNN(word2int,"lstm")
  cnn_model = makeNN(word2int,"cnn")
  lstm_results = runLSTM(lstm_model, trainingData)
  cnn_results = runLSTM(cnn_model, trainingData)
  print("Done")