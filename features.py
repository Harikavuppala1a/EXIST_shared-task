import csv
import numpy as np
import gensim.models as gsm
import time
import re,requests,json,os

def getEmojiEmbeddings(emojiList,emoji_filename,dim=300):
  """ Generates an emoji vector by averaging the emoji representation for each emoji. If no emoji returns an empty list of dimension dim"""
  e2v = gsm.KeyedVectors.load_word2vec_format(emoji_filename, binary=True)
  result = np.zeros(dim)
  if (len(emojiList) == 0):
    return result
  else:
    embs = None
    embs = np.mean([e2v[i] for i in emojiList if i in e2v.vocab], axis=0)
  if np.any(np.isnan(embs)):
    return result
  result[:300] = embs
  return result

def perspective_features(text, language,attr_dict,attributes,url):
    data_dict = {
        'comment': {'text': text},
        'languages': language,
        'requestedAttributes': attr_dict
    }
    # print (attributes)
    time.sleep(1*2)
    response = requests.post(url=url, data=json.dumps(data_dict)) 
    response_dict = json.loads(response.content)
    print (response_dict)
    # print (response_dict)
    pers_dict = {"summary": {}, "span": {}}
    for attr in attributes:
        pers_dict["summary"][attr] = response_dict["attributeScores"][attr]["summaryScore"]["value"]
        curr_span = []
        spanScores = response_dict["attributeScores"][attr]["spanScores"]
        for span in spanScores:
            curr_span.append({'begin': span['begin'], 'end': span['end'], 'score': span['score']['value']})
        pers_dict["span"][attr] = curr_span
    
    return pers_dict

def save_perspectivefeats(text,language,max_sent_cnt):
    # api_key = 'AIzaSyAoy6nY8_3SE2qS5y9OK5ggTuImPELoss0'
    api_key = 'AIzaSyBfAcfdHYFIYxCszLqn4AHwym4QXofB-eY'
    url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +'?key=' + api_key)
    if language == "en":
      attributes = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT', 'SEXUALLY_EXPLICIT', 'OBSCENE','FLIRTATION']
    else:
      attributes = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK_EXPERIMENTAL','INSULT_EXPERIMENTAL','PROFANITY_EXPERIMENTAL','THREAT_EXPERIMENTAL', ]
    attr_dict = {}
    for attr in attributes:
        attr_dict[attr] = {}  
    # perspective_feat = np.zeros(10)
    perspective_feat = np.zeros((max_sent_cnt, 10))
    l = min(max_sent_cnt,len(text))
    for in_ind,sent in enumerate(text[:l]):
      perspective_dict = perspective_features(sent, language,attr_dict,attributes,url)
      for items in perspective_dict.items():
        summary_key,summary_value = items
        for ind_summary, tup in enumerate(summary_value.items()):
          k,v = tup
          perspective_feat[in_ind, ind_summary] = v
        break
    return perspective_feat

def read_lexicon(lexicon_filename,LEN,categories):
    lexicon = dict()
    with open(lexicon_filename) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row["lemma"] in lexicon:
                lexicon[row["lemma"].strip()] = np.zeros(2*LEN)
            if row["category"] in categories:
                if row["level"] == "inclusive":
                    lexicon[row["lemma"].strip()][LEN + categories.index(row["category"])] += 1
                else:
                    lexicon[row["lemma"].strip()][categories.index(row["category"])] += 1
    return lexicon

def check_presence(text, lexicon,LEN):
    final_features = np.zeros(2*LEN)
    for k,v in lexicon.items():
        string = r"\b" + k+ r"\b"                 
        all_matches = re.findall(string, text.strip())
        for match in all_matches:
            # print (match)
            if match.strip() == "refuses" or match.strip() == "what the fuck":
                continue          
            final_features = np.add(final_features, lexicon[match.strip()])
    return final_features