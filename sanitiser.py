'''
The data set is absolute trash so we need to sanitise the data. From what we know so far, the data contains 6335 articles with each marked as FAKE (0) or REAL (1). First glimpse at the CSV file tells us that the header structure of what the csv is supposed to be like.

> ID,TEXT,LABEL

So we know that the ID comes first, so it looks like we can start with this regular expression `\n[0-9]*`, which will find the ID that predates the text. This regEx shows 6342 results so it's looking promising. Let's start by importing the file, and splitting it using this regular expression. Then we can potentially look at another expression `,[0|1]\n` which will find our classifier at the end of a container. 

It's important to note that there are articles that span several lines so we need to make sure that we don't catch false positives for our second expression.
'''

import re
from bs4 import BeautifulSoup
import json

def newDataset():
  return {
    'ID' : 0,
    'data' : "",
    'class' : 0
  }

social_media_companies = [
  "Google", "Pinterest", "Digg", "Linkedin", "Reddit", "Stumbleupon", "Print", "Delicious", "Pocket", "Tumblr"
]

redundant_text = [
"Featured image via twitter",
"share this"
"follow us on Twitter"
]

debug = False

# set up our regular expressions we'll be using
expressions = {
  'ID' : re.compile('([0-9]+,)'),
  'class' : re.compile(',(0|1)\n'),
  'url' : re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',re.IGNORECASE|re.DOTALL)
}

def cleanSentence(x):
  # remove urls
  x = expressions['url'].sub("", x)
  # # incase there are html tags, remove them.
  # x = BeautifulSoup(x, "lxml").get_text()
  #remove line breaks
  x = x.split('\n')[0]
  # remove odd unicode
  x.replace(u'\xa0', ' ').encode('utf-8')
  # remove duplicate quotes
  x.replace('""', '"')
  # remove duplicate spaces
  x = re.sub(' +',' ',x)
  # remove apostrophies
  x = re.sub("'", "", x)
  # letters and numbers only!
  x = re.sub("[^a-zA-Z0-9.]"," ", x) # The text to search
  # lowercase
  x = x.lower()
  # remove stop words
  return x


def sanitise(filename="news_ds.csv"):
  # The csv is trash so import it as a text file. I'm going to convert it into a python dictionary.
  training_set = {
    "fake" : [],
    "real" : [],
    "data" : {}
  }
  text_file = open(filename, "r")
  lines = text_file.readlines()

  # there's 166,355 lines. 
  numberOfLines = len(lines)

  entry = newDataset()

  # skip the first line; we know what it is.
  for i in range(1,len(lines)):
  # for i in range(1, 33):
    # The ID tends to be in the first 8 characters of the line
    line = lines[i]
    # Default pos of text entry is 0.
    data_startPos = 0

    # Let's find the ID
    has_ID = expressions['ID'].match(line[:10])
    if has_ID:
      # our ID is contained here, initialise a new entry.
      entry = newDataset()
      entry['ID'] = int(re.search(r'\d+', has_ID.group()).group())
      data_startPos = has_ID.end()
      if debug:
        print("Found ID: ",entry['ID'])
    
    # Find Text
    text = line[data_startPos:]
    text = cleanSentence(text)

    # Let's try to find a classifier at the end of the line.
    has_classifier = expressions['class'].search(line[-5:])
  
    if has_classifier:
      # we found a classifier, now we can finalise the dataset and make a new one!
      entry['class'] = int(re.search(r'\d+', has_classifier.group()).group())
      if debug:
        print("Found Class: ",entry['class'])
      # print(repr(entry['data']))
      index = has_classifier.span()
      su = -index[1]+index[0]
      text = text[0:len(text)+su+1]

    # print(repr(text))
    entry['data'] += text

    if has_classifier:
      if debug:
        print(entry['data'])
      training_set['data'][entry['ID']] = entry
      if entry['class'] == 0:
        training_set['fake'].append(entry['ID'])
      else:
        training_set['real'].append(entry['ID'])
      entry = newDataset()
      if debug:
        input()
        print(chr(27) + "[2J")

  text_file.close()

  # decide test and training data
  training_set['test_data'] = {}
  for i in training_set['real'] + training_set['fake']:
    training_set['test_data'][i] = False
  
  # choose the last 150 real and fake news articles as testing data.
  for i in training_set['real'][-150:] + training_set['fake'][-150:]:
    training_set['test_data'][i] = True

  return training_set

def saveJSON(dictionary):
  with open('trainingset.json', 'w') as fp:
    json.dump(dictionary, fp)
  print("saved results to trainingset.json")

if __name__ == "__main__":
  debug = False
  print("this shouldn't be called.")
  training_set = sanitise("news_ds.csv")

  print("Completed Sanitiser.")
  print("There are", len(training_set['data']), "entries.")
  print(len(training_set['fake']), "are fake")
  print(len(training_set['real']), "are real")

  saveJSON(training_set)