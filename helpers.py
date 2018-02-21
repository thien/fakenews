import json
import sanitiser as san

def loadJSON():
  training_data = {}
  try:
    print("Attempting to load trainingset.json.. ", end='')
      
    with open('trainingset.json', 'r') as fp:
      training_data = json.load(fp)
    print("Done.")
  except:
    print("Can't load trainingset.json")
    print("Creating dataset from scratch.. ", end='')
    training_data = san.sanitise("news_ds.csv")
    print("Done.")
  return training_data