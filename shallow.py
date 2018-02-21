import textacy
import numpy as np
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
    for i in text:
      if i not in frequency.keys():
        frequency[i] = 2
      else:
        frequency[i] += 1

    for i in frequency:
      frequency[i] = np.log10(frequency[i])

    dataset['data'][key]['tf'] = frequency
    dataset['data'][key]['length'] = len(text)
    
  print("Done.")
  return dataset

def df(dataset):
  print("Running Inverse Document Frequency.. ", end='')
  
  dataset['df'] = {}
  for key in dataset['data']:
    frequency = dataset['data'][key]['tf']
    for i in frequency:
      if i not in dataset['df'].keys():
        dataset['df'][i] = 2
      else:
        dataset['df'][i] += 1

  print("Done.")
  return dataset

def tfidf(dataset):
  print("Running TF-IDF.. ", end='')

  for key in dataset['data']:
    dataset['data'][key]['tfidf'] = {}
    for word in dataset['data'][key]['tf']:
      tf = dataset['data'][key]['tf'][word]
      idf = np.log10(dataset['data'][key]['length'] / dataset['df'][word])

      dataset['data'][key]['tfidf'][word] = tf * idf
    print(dataset['data'][key]['tfidf'])
    break
  print("Done.")
  return dataset

def ngram():
  return 0

# -----------------------------
# Classification
# -----------------------------

def naive_bayes():
  return 0

def calculateAccuracy():
  return 0

def calculatePrecision():
  return 0

def calculateRecall():
  return 0

def calculateF1Measure():
  return 0