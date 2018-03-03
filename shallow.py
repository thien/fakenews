# import textacy
import numpy as np
import sanitiser
# -----------------------------
# Pre Processing and Feature Extraction
# -----------------------------

def tf(dataset):
  print("Running Term-Frequency.. ", end='')
  for article in dataset['data']:
    text = dataset['data'][article]['data']
    # need to split this into separate words
    # text = text.split(" ")

    frequency = {}
    for word in text:
      if word not in frequency.keys():
        frequency[word] = 2
      else:
        frequency[word] += 1

    for word in frequency:
      frequency[word] = np.log10(frequency[word])

    dataset['data'][article]['tf'] = frequency
    dataset['data'][article]['length'] = len(text)
    
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

  # need to count the number of unique words in the class.
  numberOfWordsInFake = 0
  numberOfWordsInReal = 0

  # count the overall number of words in the class.
  sumNumberOfWordsInFake = 0
  sumNumberOfWordsInReal = 0

  for word in dataset['words']:
    if dataset['words'][word]['fake'] > 0:
      numberOfWordsInFake += 1
      sumNumberOfWordsInFake += dataset['words'][word]['fake']
    if dataset['words'][word]['real'] > 0:
      numberOfWordsInReal += 1
      sumNumberOfWordsInReal += dataset['words'][word]['real']

  dataset['statistics'] = {
    'uniqueWordsInFake' : numberOfWordsInFake,
    'uniqueWordsInReal' : numberOfWordsInReal,
    'wordsInFake' : sumNumberOfWordsInFake,
    'wordsInReal' : sumNumberOfWordsInReal
  }

  dataset['training_data'] = {
    'fake_articles' : [],
    'real_articles' : []
  }
  # count the training data for fake and real articles
  for article in dataset['data']:
    # look at the training data.
    if not dataset['test_data'][article]:
      # check if it's fake news or not and group them.
      if dataset['data'][article]['class'] == 0:
        dataset['training_data']['fake_articles'].append(article)
      else:
        dataset['training_data']['real_articles'].append(article)
    dataset['training_data']['size'] = len(dataset['training_data']['fake_articles'])
    dataset['training_data']['size'] +=len(dataset['training_data']['real_articles'])

  print("Number of Articles:", dataset['training_data']['size'])
  print("Number of Fake Articles:", len(dataset['training_data']['fake_articles']))
  print("Number of Real Articles:", len(dataset['training_data']['real_articles']))

  # calculate the probability of fake and real news based on training dataset
  p_fake_article = len(dataset['training_data']['fake_articles'])/dataset['training_data']['size']
  p_real_article = len(dataset['training_data']['real_articles'])/dataset['training_data']['size']
  dataset['training_data']['prob_fake_article'] = p_fake_article
  dataset['training_data']['prob_real_article'] = p_real_article
  print("Probability of Real Article:", p_real_article, ", Fake:", p_fake_article)
  return dataset

# Use a Multinomial Naïve Bayes classifier to discriminate fake from real news. (5 marks)
def naive_bayes(dataset):
  # take note of how well this performs.
  results = []

  # Note: we need to consider fake and real and choose the higher probability.
  print("Running Multinomial Naive Bayes Classifier.. ", end='')
  counter = 0
  for article in dataset['test_data']:
    # look at the test data.
    if dataset['test_data'][article]:
      counter += 1
      # set up the text for analysis
      article_text = dataset['data'][article]['data']
      # count the vocab size
      vocabularySize = len(article_text)

      fake_cond_probs = []
      real_cond_probs = []

      count_fake_plus_vocab = dataset['statistics']['wordsInFake'] + vocabularySize
      count_real_plus_vocab = dataset['statistics']['wordsInReal'] + vocabularySize

      # calculate the probability that it is fake news
      for word in dataset['data'][article]['tf']:
        # grab it from our list of words
        word_p = {
          'fake' : 1,
          'real' : 1,
          'df' : 1
        }
        if word in dataset['words']:
          word_p = dataset['words'][word]
          # print(word_p)
        else:
          # we just keep the word_p template.
          pass
      
        # print("Word: ", word, ", Fake Encouters:", word_p['fake'], ", Real Encounters:", word_p['real'])
        cond_prob_word_fake = ((word_p['fake']+1) / count_fake_plus_vocab)
        cond_prob_word_real = ((word_p['real']+1) / count_real_plus_vocab)
        # print("Word:", word, "p(f):", cond_prob_word_fake, "p(r):", cond_prob_word_real)
        fake_cond_probs.append(cond_prob_word_fake)
        real_cond_probs.append(cond_prob_word_real)

      # numbers are too small so we take the logarithmic value
      cond_probability_fake = np.sum(np.log10(np.array(fake_cond_probs)))
      cond_probability_fake = cond_probability_fake * dataset['training_data']['prob_fake_article']
      cond_probability_real = np.sum(np.log10(np.array(real_cond_probs)))
      cond_probability_real = cond_probability_real * dataset['training_data']['prob_real_article']
      
      # generate results container for analysis
      result = {
        'number' : counter,
        'id' : article,
        'p_fake' : cond_probability_real,
        'p_real' : cond_probability_real,
        'guess' : 0 if (cond_probability_fake > cond_probability_real) else 1,
        'actual' : dataset['data'][article]['class']
      }
      results.append(result)

  print("Done.")
  return results

# Compare the classification results using the different features you extracted in the previous step. Use the classification accuracy, Precision, Recall, and F1-measure as comparison metrics.(5 Marks)

def evaluate(results):
  # generate the 2x2 contingency table
  # alongside the accuracy
  total = len(results)
  accuracy = 0
  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0

  for i in results:
    # calculate accuracy
    if i['guess'] == i['actual']:
      accuracy += 1
    # calculate true positive
      if (i['guess'] == 1):
        true_positive += 1
      else:
        false_negative += 1
    else:
      if (i['guess'] == 1):
        false_positive += 1
      else:
        true_negative += 1
  
  return {
    'tp' : true_positive/total,
    'fp' : false_positive/total,
    'fn' : false_negative/total,
    'tn' : true_negative/total,
    'accuracy' : accuracy/total
  }


def calculateAccuracy(scores):
  # compare the test data result with it's classed result.
  print("Accuracy: ", end='')
  print(str(scores['accuracy']*100) + "%")
  return scores['accuracy']

def calculatePrecision(s):
  # TP/(TP+FN)
  # our input is the scores generated in evaluate(results)
  print("Precision: ", end='')
  precision = s['tp']/(s['tp']+s['fn'])
  print(str(precision*100) + "%")
  return precision

def calculateRecall(s):
  # TP/(TP+FP)
  print("Recall: ", end='')
  recall = s['tp']/(s['tp']+s['fp'])
  print(str(recall*100) + "%")
  return recall

def calculateF1Measure(precision, recall):
  # 2*(Precision*Recall)/(Precision+Recall)
  print("F1 Measure: ", end='')
  f1 = 2*(precision*recall)/(precision+recall)
  print(str(f1*100) + "%")
  return f1