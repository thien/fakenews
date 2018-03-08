'''
The data set is absolute trash so we need to sanitise the data. From what we know so far, the data contains 6335 articles with each marked as FAKE (0) or REAL (1). First glimpse at the CSV file tells us that the header structure of what the csv is supposed to be like.

> ID,TEXT,LABEL

So we know that the ID comes first, so it looks like we can start with this regular expression `\n[0-9]*`, which will find the ID that predates the text. This regEx shows 6342 results so it's looking promising. Let's start by importing the file, and splitting it using this regular expression. Then we can potentially look at another expression `,[0|1]\n` which will find our classifier at the end of a container. 

It's important to note that there are articles that span several lines so we need to make sure that we don't catch false positives for our second expression.
'''

import re
import json
from string import punctuation

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

# Stopwords JSON (which is much faster than a list.)
# shamelessly poached from https://raw.githubusercontent.com/6/stopwords-json/master/dist/en.json
stopwords = {"a": True, "a's": True, "able": True, "about": True, "above": True, "according": True, "accordingly": True, "across": True, "actually": True, "after": True, "afterwards": True, "again": True, "against": True, "ain't": True, "all": True, "allow": True, "allows": True, "almost": True, "alone": True, "along": True, "already": True, "also": True, "although": True, "always": True, "am": True, "among": True, "amongst": True, "an": True, "and": True, "another": True, "any": True, "anybody": True, "anyhow": True, "anyone": True, "anything": True, "anyway": True, "anyways": True, "anywhere": True, "apart": True, "appear": True, "appreciate": True, "appropriate": True, "are": True, "aren't": True, "around": True, "as": True, "aside": True, "ask": True, "asking": True, "associated": True, "at": True, "available": True, "away": True, "awfully": True, "b": True, "be": True, "became": True, "because": True, "become": True, "becomes": True, "becoming": True, "been": True, "before": True, "beforehand": True, "behind": True, "being": True, "believe": True, "below": True, "beside": True, "besides": True, "best": True, "better": True, "between": True, "beyond": True, "both": True, "brief": True, "but": True, "by": True, "c": True, "c'mon": True, "c's": True, "came": True, "can": True, "can't": True, "cannot": True, "cant": True, "cause": True, "causes": True, "certain": True, "certainly": True, "changes": True, "clearly": True, "co": True, "com": True, "come": True, "comes": True, "concerning": True, "consequently": True, "consider": True, "considering": True, "contain": True, "containing": True, "contains": True, "corresponding": True, "could": True, "couldn't": True, "course": True, "currently": True, "d": True, "definitely": True, "described": True, "despite": True, "did": True, "didn't": True, "different": True, "do": True, "does": True, "doesn't": True, "doing": True, "don't": True, "done": True, "down": True, "downwards": True, "during": True, "e": True, "each": True, "edu": True, "eg": True, "eight": True, "either": True, "else": True, "elsewhere": True, "enough": True, "entirely": True, "especially": True, "et": True, "etc": True, "even": True, "ever": True, "every": True, "everybody": True, "everyone": True, "everything": True, "everywhere": True, "ex": True, "exactly": True, "example": True, "except": True, "f": True, "far": True, "few": True, "fifth": True, "first": True, "five": True, "followed": True, "following": True, "follows": True, "for": True, "former": True, "formerly": True, "forth": True, "four": True, "from": True, "further": True, "furthermore": True, "g": True, "get": True, "gets": True, "getting": True, "given": True, "gives": True, "go": True, "goes": True, "going": True, "gone": True, "got": True, "gotten": True, "greetings": True, "h": True, "had": True, "hadn't": True, "happens": True, "hardly": True, "has": True, "hasn't": True, "have": True, "haven't": True, "having": True, "he": True, "he's": True, "hello": True, "help": True, "hence": True, "her": True, "here": True, "here's": True, "hereafter": True, "hereby": True, "herein": True, "hereupon": True, "hers": True, "herself": True, "hi": True, "him": True, "himself": True, "his": True, "hither": True, "hopefully": True, "how": True, "howbeit": True, "however": True, "i": True, "i'd": True, "i'll": True, "i'm": True, "i've": True, "ie": True, "if": True, "ignored": True, "immediate": True, "in": True, "inasmuch": True, "inc": True, "indeed": True, "indicate": True, "indicated": True, "indicates": True, "inner": True, "insofar": True, "instead": True, "into": True, "inward": True, "is": True, "isn't": True, "it": True, "it'd": True, "it'll": True, "it's": True, "its": True, "itself": True, "j": True, "just": True, "k": True, "keep": True, "keeps": True, "kept": True, "know": True, "known": True, "knows": True, "l": True, "last": True, "lately": True, "later": True, "latter": True, "latterly": True, "least": True, "less": True, "lest": True, "let": True, "let's": True, "like": True, "liked": True, "likely": True, "little": True, "look": True, "looking": True, "looks": True, "ltd": True, "m": True, "mainly": True, "many": True, "may": True, "maybe": True, "me": True, "mean": True, "meanwhile": True, "merely": True, "might": True, "more": True, "moreover": True, "most": True, "mostly": True, "much": True, "must": True, "my": True, "myself": True, "n": True, "name": True, "namely": True, "nd": True, "near": True, "nearly": True, "necessary": True, "need": True, "needs": True, "neither": True, "never": True, "nevertheless": True, "new": True, "next": True, "nine": True, "no": True, "nobody": True, "non": True, "none": True, "noone": True, "nor": True, "normally": True, "not": True, "nothing": True, "novel": True, "now": True, "nowhere": True, "o": True, "obviously": True, "of": True, "off": True, "often": True, "oh": True, "ok": True, "okay": True, "old": True, "on": True, "once": True, "one": True, "ones": True, "only": True, "onto": True, "or": True, "other": True, "others": True, "otherwise": True, "ought": True, "our": True, "ours": True, "ourselves": True, "out": True, "outside": True, "over": True, "overall": True, "own": True, "p": True, "particular": True, "particularly": True, "per": True, "perhaps": True, "placed": True, "please": True, "plus": True, "possible": True, "presumably": True, "probably": True, "provides": True, "q": True, "que": True, "quite": True, "qv": True, "r": True, "rather": True, "rd": True, "re": True, "really": True, "reasonably": True, "regarding": True, "regardless": True, "regards": True, "relatively": True, "respectively": True, "right": True, "s": True, "said": True, "same": True, "saw": True, "say": True, "saying": True, "says": True, "second": True, "secondly": True, "see": True, "seeing": True, "seem": True, "seemed": True, "seeming": True, "seems": True, "seen": True, "self": True, "selves": True, "sensible": True, "sent": True, "serious": True, "seriously": True, "seven": True, "several": True, "shall": True, "she": True, "should": True, "shouldn't": True, "since": True, "six": True, "so": True, "some": True, "somebody": True, "somehow": True, "someone": True, "something": True, "sometime": True, "sometimes": True, "somewhat": True, "somewhere": True, "soon": True, "sorry": True, "specified": True, "specify": True, "specifying": True, "still": True, "sub": True, "such": True, "sup": True, "sure": True, "t": True, "t's": True, "take": True, "taken": True, "tell": True, "tends": True, "th": True, "than": True, "thank": True, "thanks": True, "thanx": True, "that": True, "that's": True, "thats": True, "the": True, "their": True, "theirs": True, "them": True, "themselves": True, "then": True, "thence": True, "there": True, "there's": True, "thereafter": True, "thereby": True, "therefore": True, "therein": True, "theres": True, "thereupon": True, "these": True, "they": True, "they'd": True, "they'll": True, "they're": True, "they've": True, "think": True, "third": True, "this": True, "thorough": True, "thoroughly": True, "those": True, "though": True, "three": True, "through": True, "throughout": True, "thru": True, "thus": True, "to": True, "together": True, "too": True, "took": True, "toward": True, "towards": True, "tried": True, "tries": True, "truly": True, "try": True, "trying": True, "twice": True, "two": True, "u": True, "un": True, "under": True, "unfortunately": True, "unless": True, "unlikely": True, "until": True, "unto": True, "up": True, "upon": True, "us": True, "use": True, "used": True, "useful": True, "uses": True, "using": True, "usually": True, "uucp": True, "v": True, "value": True, "various": True, "very": True, "via": True, "viz": True, "vs": True, "w": True, "want": True, "wants": True, "was": True, "wasn't": True, "way": True, "we": True, "we'd": True, "we'll": True, "we're": True, "we've": True, "welcome": True, "well": True, "went": True, "were": True, "weren't": True, "what": True, "what's": True, "whatever": True, "when": True, "whence": True, "whenever": True, "where": True, "where's": True, "whereafter": True, "whereas": True, "whereby": True, "wherein": True, "whereupon": True, "wherever": True, "whether": True, "which": True, "while": True, "whither": True, "who": True, "who's": True, "whoever": True, "whole": True, "whom": True, "whose": True, "why": True, "will": True, "willing": True, "wish": True, "with": True, "within": True, "without": True, "won't": True, "wonder": True, "would": True, "wouldn't": True, "x": True, "y": True, "yes": True, "yet": True, "you": True, "you'd": True, "you'll": True, "you're": True, "you've": True, "your": True, "yours": True, "yourself": True, "yourselves": True, "z": True, "zero": True,  " ": True}

debug = False

# set up our regular expressions we'll be using
expressions = {
  'ID' : re.compile('([0-9]+,)'),
  'class' : re.compile(',(0|1)\n'),
  'url' : re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',re.IGNORECASE|re.DOTALL),
  'twittercom': re.compile('[a-zA-Z]+twittercom[a-zA-Z0-9]+')
}

def cleanSentence(x):
  # lowercase
  x = x.lower()

  x = x.replace("u.s.", "usa")

  # remove “ and ”
  x = x.replace("“", "")
  x = x.replace("”", "")

  x = x.replace(".", ". ")

  # remove twitter urls entirely..
  x = expressions['twittercom'].sub("", x)

  # if there is an apostrophe next to two letters
  # then delete everything past the apostrophe, including itself.
  x = x.replace("’s", "")
  x = x.replace("’", "")
  x = x.replace("'s", "")

  # remove urls
  x = expressions['url'].sub("", x)
  #remove line breaks
  x = x.split('\n')[0]
  # remove odd unicode
  x.replace(u'\xa0', ' ').encode('utf-8')
  # remove duplicate quotes
  x.replace('""', '"')
  x.replace('"', ' ')
  # remove apostrophies
  x = re.sub("'", " ", x)

  # letters and numbers only!
  x = re.sub("[^a-zA-Z0-9.]"," ", x) # The text to search
  # remove punctuation
  x = ''.join(c for c in x if c not in punctuation)
  return x

def cleanArticle(x):
  # split words prior to submission
  x = x.split(" ")
  # remove empty strings
  x = list(filter(None, x))
  # remove stopwords
  x = [word.rstrip('.') for word in x if word not in stopwords.keys()]
  # return list of words
  return x

def sanitise(filename="news_ds.csv", numberOfTestData=150):
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
    if i % 300 == 0:
      print("Creating sanitised dataset from scratch.. Loading: " + str(round((i/len(lines)*100),2)) + "%\r", end="" )

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
      entry['data'] = cleanArticle(entry['data'])
      training_set['data'][entry['ID']] = entry
      if entry['class'] == 0:
        training_set['fake'].append(entry['ID'])
      else:
        training_set['real'].append(entry['ID'])
      entry = newDataset()
      if debug:
        input()
        print(chr(27) + "[2J")
  print("Creating sanitised dataset from scratch.. Loading: 100.0% ",end="")  
  text_file.close()

  # decide test and training data
  training_set['test_data'] = {}
  for i in training_set['real'] + training_set['fake']:
    training_set['test_data'][i] = False
  
  # choose the last 150 real and fake news articles as testing data.
  for i in training_set['real'][-numberOfTestData:] + training_set['fake'][-numberOfTestData:]:
    training_set['test_data'][i] = True

  return training_set

if __name__ == "__main__":
  import helpers
  debug = False
  print("Sanitising data.. ")
  training_set = sanitise("news_ds.csv")

  print("There are", len(training_set['data']), "entries.")
  print(len(training_set['fake']), "are fake")
  print(len(training_set['real']), "are real")

  helpers.saveJSON(training_set, "trainingset.json")