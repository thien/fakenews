# import textacy
import numpy as np
import sanitiser
from nltk import trigrams, bigrams

"""
For shallow, 
"""

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
            'tfidf': 0,
            'df': 1
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
  for article in dataset['data']:
    # check if the article is training data or not before
    # continuing (if it's marked as a 1 then it is test data.)
    # if not dataset['test_data'][article]:
    tfidf_set = {}
    for word in dataset['data'][article]['tf']:
      # print(word)
      tf = dataset['data'][article]['tf'][word]
      document_length = dataset['data'][article]['length'] 
      
      if word in dataset['words']:
        word_document_frequency = dataset['words'][word]['df']
        # idf = np.log10(document_length / word_document_frequency)
        idf = (document_length / word_document_frequency)
        # add it to the list of tfidf words
        tfidf_set[word] = tf * idf
    dataset['data'][article]['tfidf'] = tfidf_set
  print("Done.")
  return dataset

def ngram(dataset, n=3):
  print("Calculating N-Gram Feature Vector.. ", end="")
  # i'd imagine that having an efficent method to compute ngrams would
  # be very difficult given the span of time allocated for the assignment,
  # so i'll be using nltk.

  ngramFeatureVector = {}
  unique_fake = 0
  unique_real = 0
  fake = 0
  real = 0
  for article in dataset['data']:
    data = dataset['data'][article]['data']
    grams = None
    # awkwardly we cant iterate through it after this function
    # so this is a new variable that's more universally accessible
    # (we'll need this for our naive bayes)
    gramSet = []
    if n == 3:
      grams = trigrams(data)
    elif n == 2:
      grams = bigrams(data)
    # iterate through the grams generated
    local_count = 0
    for gram in grams:
      local_count += 1
      # create a new string which will be our key
      g = ' '.join(gram)
      gramSet.append(g)
      if g not in ngramFeatureVector:
        ngramFeatureVector[g] = {
          "fake": 0,
          "real": 0
        }
        # count the unique entry (which will be the first entry)
        if dataset['data'][article]['class'] == 1:
          unique_real += 1
        else:
          unique_fake += 1
      # now accumulate the frequency of the feature vect
      if dataset['data'][article]['class'] == 1:
        ngramFeatureVector[g]['real'] += 1
        real += 1
      else:
        ngramFeatureVector[g]['fake'] += 1
        fake += 1

    # update information for our article data
    dataset['data'][article]['ngram'] = {
      n : {
        "size" : local_count,
        "grams" : gramSet
      }
    }
  
  dataset['ngrams'] = {
    n : {
      "size" : {
          "overall": fake + real,
          "fake" : fake,
          "real" : real
      },
      "unique_size" : {
        "overall" : len(ngramFeatureVector),
        "fake" : unique_fake,
        "real" : unique_real
      },
      "grams": ngramFeatureVector
    }
  }
  print("Done.")
  return dataset

# -----------------------------
# Classification
# -----------------------------

def preprocess_probabilities(dataset, debug=False):
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

  # calculate the probability of fake and real news based on training dataset
  p_fake_article = len(dataset['training_data']['fake_articles'])/dataset['training_data']['size']
  p_real_article = len(dataset['training_data']['real_articles'])/dataset['training_data']['size']
  dataset['training_data']['prob_fake_article'] = p_fake_article
  dataset['training_data']['prob_real_article'] = p_real_article

  if debug:
    print("Number of Articles:", dataset['training_data']['size'])
    print("Number of Fake Articles:", len(dataset['training_data']['fake_articles']))
    print("Number of Real Articles:", len(dataset['training_data']['real_articles']))
    print("Probability of Real Article:", p_real_article, ", Fake:", p_fake_article)
  return dataset

# Use a Multinomial Naïve Bayes classifier to discriminate fake from real news. (5 marks)
def naive_bayes(dataset, option="tf", ngramSize=3):
  # take note of how well this performs.
  results = []

  # Note: we need to consider fake and real and choose the higher probability.
  print("Running Multinomial Naive Bayes Classifier (via "+option+").. ", end='')
  # print()
  counter = 0
  for article in dataset['test_data']:
    # check if this article is a test_data
    if dataset['test_data'][article]:
      counter += 1
      # set up the text for analysis
      article_text = dataset['data'][article]['data']
      # count the vocab size
      vocabularySize = len(article_text)

      fake_cond_probs,real_cond_probs = [], []

      count_fake_plus_vocab = dataset['statistics']['wordsInFake'] + vocabularySize
      count_real_plus_vocab = dataset['statistics']['wordsInReal'] + vocabularySize

      cond_probability_fake, cond_probability_real = None, None

      # process ngram option
      if option == "ngrams":
        # need to count the total of fake and real ngrams + the range of vocab
        count_fake_ngram = dataset['ngrams'][ngramSize]['size']['fake']
        count_real_ngram = dataset['ngrams'][ngramSize]['size']['real']
        ngram_vocab = dataset['ngrams'][ngramSize]['unique_size']['overall']
        # print(dataset['data'][article]['ngram'][ngramSize]['grams'])
        # print("looking at ngrams")
        # print(dataset['data'][article]['ngram'][ngramSize])
        for gram in dataset['data'][article]['ngram'][ngramSize]['grams']:
          # print(gram)
          # print(dataset['ngrams'][ngramSize])
          gram_p = {
            'fake' : 1,
            'real' : 1
          }
          # check if this gram exists, we can't make entries for non-existent ngrams in the corpus since there is a very high chance that it isn't in the corpus, thus flooding the classifier with not very useful information. 
          if gram in dataset['ngrams'][ngramSize]['grams'].keys():
            gram_p = dataset['ngrams'][ngramSize]['grams'][gram]
          # print("Gram: ", gram, ", Fake Encouters:", gram_p['fake'], ", Real Encounters:", gram_p['real'])
          # now we need to count the conditional probability
          cond_prob_gram_fake = (gram_p['fake']+1)/(count_fake_ngram+ngram_vocab)
          cond_prob_gram_real = (gram_p['real']+1)/(count_real_ngram+ngram_vocab)
          # print("\t", gram, "p(f):", cond_prob_gram_fake, "p(r):", cond_prob_gram_real)
          fake_cond_probs.append(cond_prob_gram_fake)
          real_cond_probs.append(cond_prob_gram_real)
        cond_probability_fake = np.sum(np.log10(np.array(fake_cond_probs)))
        cond_probability_real = np.sum(np.log10(np.array(real_cond_probs)))
        # print("fake:", cond_probability_fake, "real:", cond_probability_real)
      # process tf option
      if option == "tf":
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
          # P(word|class)=(word_count_in_class + 1)/(total_words_in_class+total_unique_words_in_class) 

          # print(cond_prob_word_fake)
          # print(cond_prob_word_real)
          # print("Word:", word, "p(f):", cond_prob_word_fake, "p(r):", cond_prob_word_real)
          fake_cond_probs.append(cond_prob_word_fake)
          real_cond_probs.append(cond_prob_word_real)
        # print(fake_cond_probs)
        # print(real_cond_probs)
        cond_probability_fake = np.sum(np.log10(np.array(fake_cond_probs)))
        cond_probability_real = np.sum(np.log10(np.array(real_cond_probs)))
      elif option == "tfidf":
        # we need to compute tfidf against the test data.
        # for each test document, we compare the word's frequency against the inverse document frequencies of our training data.
        document_length = dataset['data'][article]['length'] 
        numberOfWords = len(dataset['words'])
        # print(dataset['data'][article])
        for word in dataset['data'][article]['tf']:
          # grab it from our list of words
          word_p = {
            'fake' : 1,
            'real' : 1,
            'df' : 1
          }
          # only consider words that we have in our training data.
          if word in dataset['words']:
            word_p = dataset['words'][word]
            # tf = dataset['data'][article]['tf'][word]

            # calculate frequency of the word appearing in fake news and real news
            # just to make sure we don't divide by zero we add a +1
            document_frequency_fake = word_p['fake']+1
            document_frequency_real = word_p['real']+1
            # # calculate tfidf for fake and real datasets
            # fake_tfidf  = tf * (1 / document_frequency_fake)
            # real_tfidf  = tf * (1 / document_frequency_real)

            fake_tfidf = document_frequency_fake * (numberOfWords/word_p['df'])
            real_tfidf = document_frequency_real * (numberOfWords/word_p['df'])

            # print(word, "Fake:", fake_tfidf, "Real:", real_tfidf)
            # fake_tfidf, real_tfidf = np.log10(fake_tfidf), np.log10(real_tfidf)
            # print("\t", "logarithmic:", fake_tfidf, real_tfidf)
            # calculate the conditional probability
            cond_prob_word_fake = (fake_tfidf) / count_fake_plus_vocab
            cond_prob_word_real = (real_tfidf) / count_real_plus_vocab
            # P(word|class)=(tfidf_of_word + 1)/(total_words_in_class+total_unique_words_in_class) 

            # print("cond_prob of word in fake and real doc",cond_prob_word_fake, cond_prob_word_real)
            # print("Word:", word, "p(f):", cond_prob_word_fake, "p(r):", cond_prob_word_real)
            fake_cond_probs.append(cond_prob_word_fake)
            real_cond_probs.append(cond_prob_word_real)
        # values are so small we need to consider it in logarithmic form.
        cond_probability_fake = np.sum(np.log10(np.array(fake_cond_probs)))
        cond_probability_real = np.sum(np.log10(np.array(real_cond_probs)))
        # cond_probability_fake = np.prod(np.array(fake_cond_probs))
        # cond_probability_real = np.prod(np.array(real_cond_probs))
        # cond_probability_fake = np.sum(np.array(fake_cond_probs))
        # cond_probability_real = np.sum(np.array(real_cond_probs))

      # print("Sum of cond probs:", cond_prob_word_fake, cond_prob_word_real)
      cond_probability_fake = cond_probability_fake * dataset['training_data']['prob_fake_article']
      cond_probability_real = cond_probability_real * dataset['training_data']['prob_real_article']
      # print("Sum of cond probs Normalised:", cond_prob_word_fake, cond_prob_word_real)
      # generate results container for analysis
      result = {
        'number' : counter,
        'id' : article,
        'p_fake' : cond_probability_fake,
        'p_real' : cond_probability_real,
        'guess' : 0 if (cond_probability_fake > cond_probability_real) else 1,
        'actual' : dataset['data'][article]['class']
      }
      # print(result)
      results.append(result) 
      # break
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
  
  # collate results as percentages
  s =  {
    'tp' : true_positive/total,
    'fp' : false_positive/total,
    'fn' : false_negative/total,
    'tn' : true_negative/total,
    'accuracy' : accuracy/total
  }
  
  # calculate precision:  TP/(TP+FN)
  s['precision'] = s['tp']/(s['tp']+s['fn'])
  # calculate recall: TP/(TP+FP)
  s['recall'] = s['tp']/(s['tp']+s['fp'])
  # calculate F1 Measure: 2*(precision*recall)/(precision+recall)
  s['f1_measure'] = 2*(s['precision']*s['recall'])/(s['precision']+s['recall'])

  for i in s:
    # show each percentage in terms of 0-100% and 3 decimal places
    s[i] = round(s[i] * 100,3)
  return s

if __name__ == "__main__":
  import helpers
  dataset = helpers.loadJSON()
  dataset = tf(dataset)
  dataset = df(dataset)
  dataset = tfidf(dataset)
  gramSize = 2
  dataset = ngram(dataset, gramSize)
  dataset = preprocess_probabilities(dataset)
  # calculate naive bayes for our three methods
  # results_tf = naive_bayes(dataset, "tf")
  results_tfidf = naive_bayes(dataset, "tfidf")
  # results_ngrams = naive_bayes(dataset, "ngrams", gramSize)

  # tf_scores = evaluate(results_tf)
  tfidf_scores = evaluate(results_tfidf)
  # ngram_scores = evaluate(results_ngrams)
  # print("tf:\t", tf_scores)
  print("tfidf:\t", tfidf_scores)
  # print("ngrams:\t", ngram_scores)