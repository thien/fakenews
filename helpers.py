import json
import os
import sanitiser as san
import numpy as np
import pickle

datapath = 'datafiles'
datapath = os.path.join(os.getcwd(), datapath)

def loadJSON(trainingFile='trainingset.json'):
  training_data = {}
  try:
    print("Attempting to load '"+trainingFile+"'.. ", end='')
      
    with open(os.path.join(datapath,trainingFile), 'r') as fp:
      training_data = json.load(fp)
    print("Done.")
  except:
    print("Can't load '"+trainingFile+"'.")
    print("Creating dataset from scratch.. ", end='')
    training_data = san.sanitise("news_ds.csv")
    print("Done.")
  return training_data

def saveJSON(dictionary, name='trainingset.json'):
  filepath = os.path.join(datapath, name)
  with open(filepath, 'w') as fp:
    json.dump(dictionary, fp)
  print("saved results to", name)

def verifyChecksum(fname):
  import hashlib
  hash_md5 = hashlib.md5()
  with open(fname, "rb") as f:
      for chunk in iter(lambda: f.read(4096), b""):
          hash_md5.update(chunk)
  return hash_md5.hexdigest()


def downloadGloveDataset():
  """
  This function downloads the glove dataset to the /datafiles
  directory, and automatically extracts the dataset for use
  with the deep nets!

  Pretty cool, huh?

  Hopefully this can reduce the amount of headache involved
  in terms of downloading random junk for marking summatives
  """
  import zipfile
  import urllib.request
  dataset_name = 'glove.6B.50d.txt'

  zip_checksum = "056ea991adb4740ac6bf1b6d9b50408b"
  text_checksum = "0fac3659c38a4c0e9432fe603de60b12"

  # check if the datapath exists
  dir_exist = os.path.isdir(datapath)
  # if it doesn't then make that folder
  if not dir_exist:
    os.makedirs(datapath)

  glove_file = os.path.join(datapath, dataset_name)
  file_zip = os.path.join(datapath, "glove.6B.zip")
  # we'll use the wikipedia sample from Stanford's Glove Dataset
  # shamelessly nicked from https://nlp.stanford.edu/projects/glove/
  glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'


  # verify the md5 checksum of the file so we know the zip file is inteact.
  zip_verified = False
  if os.path.exists(file_zip):
    print(file_zip, "found, verifying checksum..")
    if verifyChecksum(file_zip) == zip_checksum:
      zip_verified = True
    else:
      print(file_zip, "is corrupt, redownloading..")

  if not zip_verified:
    print("Downloading the glove dataset (This may take a while since it's about 862.2MB..)", end="")
    import shutil
    with urllib.request.urlopen(glove_url) as response, open(file_zip, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("Done.")

  # verify that the md5 checksum of the dataset is real.
  glove_verified = False
  if os.path.exists(glove_file):
     if verifyChecksum(glove_file) == text_checksum:
      glove_verified = True
      print("Glove dataset is ready for use.")
      
  if not glove_verified:
    print("Extracting the glove dataset.. ")
    if not os.path.exists(glove_file):
      # open the zip file and extract the glove dataset
      fh = open(file_zip, 'rb')
      z = zipfile.ZipFile(fh)
      for name in z.namelist():
        if str(name) == dataset_name:
          z.extract(name, datapath)
          break
      fh.close()
      print("Done.")


def loadGlove(trainingFile='glove.pickle'):
  glove = {}
  trainingFile = os.path.join(datapath, trainingFile)
  try:
    print("Attempting to load '"+trainingFile+"'.. ", end='')
    with open(trainingFile, 'rb') as fp:
      glove = pickle.load(fp)
    print("Done.")
  except:
    print("Can't load '"+trainingFile+"'.")
    print("Creating dataset from scratch.. ", end='')
    glove = initialiseGlove()
    print("Done.")
  return glove


def initialiseGlove(file="glove.6B.50d.txt"):
  # Here we're using a Glove word2vec model! I tried using SpaCy's english model but it's rather abstract
  print("Loading Glove dataset.. ", end="")
  lines = None
  file = os.path.join(datapath, file)
  with open(file,'rb') as f:
    lines = f.readlines()

  # initialise a matrix that considers the size of our glove vector
  weights = np.zeros((len(lines), 50))
  # create a list of words that we'll append to 
  words = []
  for i,line in enumerate(lines):
    word_weights = line.split()
    words.append(word_weights[0])
    weight = word_weights[1:]
    weights[i] = np.array([float(w) for w in weight])
  # make words utf friendly
  word_vocab = [w.decode("utf-8") for w in words]
  glove = dict(zip(word_vocab, weights))
  # save this to json so we don't have to faff about generating it again which would take much longer to compile
  print("Done.")
  # since we're using NP arrays, we can't just slap it into a json file as a cache unfortunately. Pickle is used instead
  with open(os.path.join(datapath,'glove.pickle'), 'wb') as handle:
    pickle.dump(glove, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return glove


def countLongestArticle(dataset):
  longest = 0
  for article in dataset['data']:
    leng = len(dataset['data'][article]['data'])
    if leng > longest:
      longest = leng
      # print("New Max:", leng)
  return longest