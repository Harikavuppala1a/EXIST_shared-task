import requests as r
import json
import re
from collections import defaultdict
import time
import csv

class Translator_Api():
    # https://docs.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-reference
    def __init__(self):
        self.headers = {
                "Ocp-Apim-Subscription-Key":"f90f1f0ddaae4c5ca6032bd9ce6852d3", 
                "Content-Type":"application/json",
                "Ocp-Apim-Subscription-Region":"centralindia"
            }
        self.transliterate_url = "https://api.cognitive.microsofttranslator.com/transliterate?api-version=3.0&fromScript=English&toScript=Spanish"
        self.translate_url = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=en&from=es"
    
    def get_translation(self,data):
        if data == "" : return data
        data = [{"Text":data}]
        res = r.post( self.translate_url , json = data , headers = self.headers).text
        res = json.loads(res)
        print (res)
        res = [ret['translations'][0]['text'] for ret in res ]
        # print (res[0])
        return res[0]
        
    def get_transliteration(self,data):
        if data == "" : return data
        data = [{"Text":data}]
        res = r.post( self.transliterate_url , json = data , headers = self.headers).text
        res = json.loads(res)
        # print (res)
        res = [ret['text'] for ret in res]
        return res[0]


translator = Translator_Api()
f_output = open('exist_post_converted_testing.tsv', 'w', newline='')
tsv_output = csv.writer(f_output, delimiter='\t')
tsv_output.writerow(['test_case','id','source', 'language', 'text', 'task1', 'task2'])
    

with open('data/EXIST2021_test_labeled.tsv', 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter = '\t')
    count = 0
    for row in reader:
      count = count +1
      print (count)
      if row['language'] == "es":
        post_unlab = str(row['text'])
        row['language'] = "en"
        text = translator.get_translation(post_unlab)
        row['text'] = text
        tsv_output.writerow([row['test_case'],row['id'],row['source'],row['language'],row['text'], row['task1'],row['task2']])

      else:
        tsv_output.writerow([row['test_case'],row['id'],row['source'],row['language'],row['text'], row['task1'],row['task2']])
