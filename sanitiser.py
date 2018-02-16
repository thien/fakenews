'''
The data set is absolute trash so we need to sanitise the data. From what we know so far, the data contains 6335 articles with each marked as FAKE (0) or REAL (1). First glimpse at the CSV file tells us that the header structure of what the csv is supposed to be like.

> ID,TEXT,LABEL

So we know that the ID comes first, so it looks like we can start with this regular expression `\n[0-9]*`, which will find the ID that predates the text. This regEx shows 6342 results so it's looking promising. Let's start by importing the file, and splitting it using this regular expression. Then we can potentially look at another expression `,[0|1]\n` which will find our classifier at the end of a container. 

It's important to note that there are articles that span several lines so we need to make sure that we don't catch false positives for our second expression.
'''

import re

def newDataset():
  return {
    'ID' : 0,
    'data' : "",
    'class' : 0
  }

# SETUP

# set up our regular expressions we'll be using
expressions = {
  'ID' : re.compile('([0-9]+,)'),
  'class' : re.compile(',(0|1)\n')
}

# The csv is trash so import it as a text file. I'm going to convert it into a python dictionary.
training_set = []

text_file = open("news_ds.csv", "r")
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
    print("Found ID: ",entry['ID'])
  
  # Find Text
  text = line[data_startPos:]
  text = text.split('\n')[0] #remove line breaks

  # Let's try to find a classifier at the end of the line.
  has_classifier = expressions['class'].search(line[-5:])
  # print(has_classifier)
  if has_classifier:
    # we found a classifier, now we can finalise the dataset and make a new one!
    entry['class'] = int(re.search(r'\d+', has_classifier.group()).group())
    print("Found Class: ",entry['class'])
    # print(repr(entry['data']))
    index = has_classifier.span()
    su = -index[1]+index[0]
    text = text[0:len(text)+su+1]
  else:
    # if we're not at the end of the data, then just remove the line break and continue.
    text.replace("\n", "")
  
  # print(repr(text))
  entry['data'] += text

  if has_classifier:
    print(entry)
    training_set.append(entry)
    entry = newDataset()
    input()
    print(chr(27) + "[2J")

text_file.close()
