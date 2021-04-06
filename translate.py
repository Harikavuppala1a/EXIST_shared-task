import requests as r
import json
import re
from collections import defaultdict
import time



class Translator_Api():
    # https://docs.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-reference
    def __init__(self):
        self.headers = {
                "Ocp-Apim-Subscription-Key":"f90f1f0ddaae4c5ca6032bd9ce6852d3", 
                "Content-Type":"application/json",
                "Ocp-Apim-Subscription-Region":"centralindia"
            }
        self.transliterate_url = "https://api.cognitive.microsofttranslator.com/transliterate?api-version=3.0&fromScript=English&toScript=Spanish"
        self.translate_url = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=es&from=en"
    
    def get_translation(self,data):
        if data == "" : return data
        data = [{"Text":data}]
        res = r.post( self.translate_url , json = data , headers = self.headers).text
        res = json.loads(res)
        # print (res)
        res = [ret['translations'][0]['text'] for ret in res ]
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
wfile = open('expanded_unlabdata.txt', 'a')

with open('combined_unlab_bertdata.txt', 'r') as unlabfile:
    reader_unlab = unlabfile.readlines()
    count = 0
    for row_unlab in reader_unlab:
        post_unlab = str(row_unlab)
        count = count +1
        # if count == 10000:
        #     time.sleep(5*60)
        #     count = 0
        if count <= 10166:
            continue
        print (count)
        try:
            data = translator.get_translation(post_unlab)
        except:
            continue
        print (data)
        # print (data)
        wfile.write("%s" % data)
