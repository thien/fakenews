import json
import sanitiser as san

def loadJSON(trainingFile='trainingset.json'):
  training_data = {}
  try:
    print("Attempting to load '"+trainingFile+"'.. ", end='')
      
    with open(trainingFile, 'r') as fp:
      training_data = json.load(fp)
    print("Done.")
  except:
    print("Can't load '"+trainingFile+"'.")
    print("Creating dataset from scratch.. ", end='')
    training_data = san.sanitise("news_ds.csv")
    print("Done.")
  return training_data