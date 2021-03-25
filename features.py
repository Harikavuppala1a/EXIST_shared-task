from bert_serving.client import ConcurrentBertClient
import h5py
import gensim.models as gsm
import time, requests, json
import numpy as np
from empath import Empath
import os
import csv
import re




def getEmojiEmbeddings(emojiList,dim=300):
  e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
  """ Generates an emoji vector by averaging the emoji representation for each emoji. If no emoji returns an empty list of dimension dim"""
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

def save_emojifeats(data_dict,s_filename):
    emoji_feats = np.asarray([getEmojiEmbeddings(i) for i in (data_dict['emojis'])])
    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=emoji_feats)

def save_hashfeats(data_dict,s_filename):
    bc = ConcurrentBertClient()
    seg_hashtag =[]
    for hashtag in data_dict['segmented_hashtags']:
        seg_hashtag.append(' '.join(hashtag))
    
    hash_feat = np.zeros((len(seg_hashtag), 768))
    for ind in range(len(seg_hashtag)):
        if seg_hashtag[ind].strip() != '':
            hash_feat[ind] = bc.encode([seg_hashtag[ind].strip()])
    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=hash_feat)

def save_empathfeats(data_dict,s_filename):
    lexicon = Empath() 
    empath_feats_list = ['sexism','violence', 'money', 'valuable', 'domestic_work', 'hate', 'aggression', 'anticipation', 'crime', 'weakness', 'horror', 'swearing_terms', 'kill', 'sexual', 'cooking', 'exasperation', 'body', 'ridicule', 'disgust', 'anger', 'rage']
    empath_feat = np.zeros((len(data_dict['raw_tweet_texts']),len(empath_feats_list)))
    for ind in range(len(data_dict['raw_tweet_texts'])):
      empath_dict = lexicon.analyze(data_dict['raw_tweet_texts'][ind],categories = empath_feats_list,normalize=True)
      for di_ind, tup in enumerate(empath_dict.items()):
        k,v = tup
        empath_feat[ind, di_ind] = v
    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=empath_feat)

def perspective_features(text, language,en_attr_dict,en_attributes,es_attr_dict,es_attributes,url):
    if language == 'en':
        attr_dict = en_attr_dict
        attributes = en_attributes
    else:
        attr_dict = es_attr_dict
        attributes = es_attributes
    data_dict = {
        'comment': {'text': text},
        'languages': language,
        'requestedAttributes': attr_dict
    }
    # print (attributes)
    time.sleep(1*5)
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

def save_perspectivefeats(data_dict,s_filename):
    api_key = 'AIzaSyBfAcfdHYFIYxCszLqn4AHwym4QXofB-eY'
    url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +'?key=' + api_key)
    en_attributes = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT', 'TOXICITY_FAST', 'SEXUALLY_EXPLICIT', 'OBSCENE','FLIRTATION']
    es_attributes = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK_EXPERIMENTAL','INSULT_EXPERIMENTAL','PROFANITY_EXPERIMENTAL','THREAT_EXPERIMENTAL', ]
    en_attr_dict = {}
    for attr in en_attributes:
        en_attr_dict[attr] = {}   
    es_attr_dict = {}
    for attr in es_attributes:
        es_attr_dict[attr] = {}
    perspective_feat = np.zeros((len(data_dict['raw_tweet_texts']),len(en_attributes)))
    for ind in range(len(data_dict['raw_tweet_texts'])):
      print (ind)
      perspective_dict = perspective_features(data_dict['raw_tweet_texts'][ind], data_dict['language'][ind],en_attr_dict,en_attributes,es_attr_dict,es_attributes,url)
      for items in perspective_dict.items():
        summary_key,summary_value = items
        for ind_summary, tup in enumerate(summary_value.items()):
          k,v = tup
          perspective_feat[ind, ind_summary] = v
        break

    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=perspective_feat)

def read_lexicon(lexicon_filename,LEN,categories):
    lexicon = dict()
    with open("data/" + lexicon_filename) as f:
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
        
def save_hurtlexfeats(LEN,categories,s_filename,data_dict):
    hurtlex_feat = np.zeros((len(data_dict['raw_tweet_texts']),2*LEN))
    lexicon_en = read_lexicon('hurtlex_en.tsv',LEN,categories)
    lexicon_es = read_lexicon('hurtlex_es.tsv', LEN, categories)
    for ind in range(len(data_dict['raw_tweet_texts'])):
        print (ind)
        # print (data_dict['raw_tweet_texts'][ind])
        if data_dict['language'][ind] == "en":
            feat_array = check_presence(data_dict['raw_tweet_texts'][ind].lower(), lexicon_en,LEN)
        else:
            feat_array = check_presence(data_dict['raw_tweet_texts'][ind].lower(), lexicon_es,LEN)
        hurtlex_feat[ind] = feat_array/len(data_dict['raw_tweet_texts'][ind].split())
    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=hurtlex_feat)


