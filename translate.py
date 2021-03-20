import requests as r
import json
import re
from collections import defaultdict



class Translator_Api():
    # https://docs.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-reference
    def __init__(self):
        self.headers = {
                "Ocp-Apim-Subscription-Key":"e669322a133044489e6dc4e9cde3edee",
                "Content-Type":"application/json",
                "Ocp-Apim-Subscription-Region":"centralus"
            }
        self.transliterate_url = "https://api.cognitive.microsofttranslator.com/transliterate?api-version=3.0&language=es&fromScript=english&toScript=spanish"
        self.translate_url = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=en&from=en"
    
    def get_translation(self,data):
        if data == "" : return data
        data = [{"Text":data}]
        res = r.post( self.translate_url , json = data , headers = self.headers).text
        res = json.loads(res)
        res = [ ret['translations'][0]['text'] for ret in res ]
        return res[0]
        
    def get_transliteration(self,data):
        if data == "" : return data
        data = [{"Text":data}]
        res = r.post( self.transliterate_url , json = data , headers = self.headers).text
        res = json.loads(res)
        res = [ret['text'] for ret in res]
        return res[0]


translator = Translator_Api()
# translator.get_translation
# # from googletrans import Translator
# import goslate
wfile = open('expanded_unlabdata.txt', 'w')
# # total_data =[]
# gs = goslate.Goslate()
with open('combined_unlab_bertdata.txt', 'r') as unlabfile:
    reader_unlab = unlabfile.readlines()
    for row_unlab in reader_unlab:
        post_unlab = str(row_unlab)
        # if translator.detect(post_unlab) == "en":
        #     print ("hgjgjg")
        data = translator.get_transliteration(post_unlab)
        print (data)
        wfile.write("%s\n" % data)

# translator = Translator()
# result = translator.translate('Mikä on nimesi', src='fi', dest='fr')

# # print(result.src)
# # print(result.dest)
# # print(result.text)

# # translator = Translator()  # initalize the Translator object
# # translations = translator.translate(['see if this helps', 'tarun'], dest='hi')  # translate two phrases to Hindi
# # for translation in translations:  # print every translation
# #     print(translation.text)
# # import goslate
# # kd = ['see if this helps', 'i am good','tarun']
# # gs = goslate.Goslate()
# # for k in kd:
# #     print(gs.translate(k, 'de'))
