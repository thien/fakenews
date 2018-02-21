# import textacy
import numpy as np
import sanitiser
# -----------------------------
# Pre Processing and Feature Extraction
# -----------------------------

def tf(dataset):
  print("Running Term-Frequency.. ", end='')
  for key in dataset['data']:
    text = dataset['data'][key]['data']
    # need to split this into separate words
    text = text.split(" ")

    frequency = {}
    for word in text:
      if word not in frequency.keys():
        frequency[word] = 2
      else:
        frequency[word] += 1

    for word in frequency:
      frequency[word] = np.log10(frequency[word])

    dataset['data'][key]['tf'] = frequency
    dataset['data'][key]['length'] = len(text)
    
  print("Done.")
  return dataset

def df(dataset):
  print("Running Document Frequency.. ", end='')
  
  # dataset['df'] = {}
  dataset['words'] = {}
  for article in dataset['data']:
    # check if the article is training data or not before
    # continuing
    if not dataset['test_data'][article]:
      frequency = dataset['data'][article]['tf']
      for word in frequency:
        if word not in dataset['words'].keys():
          dataset['words'][word] = {
            'fake': 0,
            'real': 0,
            'df' : 1
          }
          # add bias!
          dataset['words'][word]['df'] += 1
        else:
          dataset['words'][word]['df'] += 1
        # add class identifiers for individual words
        # we'll need this for our bayes classifier.
        if dataset['data'][article]['class'] == 0:
          dataset['words'][word]['fake'] += 1
        else:
          dataset['words'][word]['real'] += 1
  print("Done.")
  return dataset

def tfidf(dataset):
  print("Running TF-IDF.. ", end='')
  for key in dataset['data']:
    # check if the article is training data or not before
    # continuing
    if not dataset['test_data'][key]:
      dataset['data'][key]['tfidf'] = {}
      for word in dataset['data'][key]['tf']:
        tf = dataset['data'][key]['tf'][word]
        idf = np.log10(dataset['data'][key]['length'] / dataset['words'][word]['df'])

      dataset['data'][key]['tfidf'][word] = tf * idf
  print("Done.")
  return dataset

def ngram():
  return 0

# -----------------------------
# Classification
# -----------------------------

def preprocess_probabilities(dataset):
  for word in dataset['words']:
    dataset['words'][word]['fake_p'] = dataset['words'][word]['fake']/len(dataset['fake'])
    dataset['words'][word]['real_p'] = dataset['words'][word]['real']/len(dataset['real'])
  return dataset

# Use a Multinomial Naïve Bayes classifier to discriminate fake from real news. (5 marks)
def naive_bayes(dataset):
  # start by cleaning the test data
  # sample_sentence = sanitiser.cleanSentence(sample_sentence)
  # we need to consider fake and real and choose the higher probability.
  # also need to consider making our sanitiser function work for any sentence.
  
  return dataset

# Compare the classification results using the different features you extracted in the previous step. Use the classification accuracy, Precision, Recall, and F1-measure as comparison metrics.(5 Marks)
def calculateAccuracy(dataset):
  return dataset

def calculatePrecision(dataset):
  return dataset

def calculateRecall(dataset):
  return dataset

def calculateF1Measure(dataset):
  return dataset