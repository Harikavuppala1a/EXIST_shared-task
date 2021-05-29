# -*- coding: utf-8 -*-
import re
import pickle
from ekphrasis.classes.segmenter import Segmenter
seg_tw = Segmenter(corpus = "twitter")
import preprocessor as tweet_proc
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import csv
from sklearn.utils import shuffle
import nltk
nltk.download('punkt')
from nltk import sent_tokenize

conf_dict_list, conf_dict_com = load_config(sys.argv[1])

def make_list(proc_obj):
  if proc_obj == None:
    return []
  store = []
  for unit in proc_obj:
    store.append(unit.match.strip())
  return store

def emotext(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
    return text

tweets = []
simple_cleaning_texts = []
tweetproc_clean_texts = []
tokenized_tweets = []
sent_list = []

ID =[]
source = []
language = []
test_case = []
task_1_labels = []
task_2_labels = []

hashtags = []
segmented_hashtags = []
emoji_texts = []
smileys = []
emojis = []
urls = []
mentions = []
numbers = []
reserveds = []

# filename = "./drive/My Drive/exist/exist_converted_test.tsv"
r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
r_white = re.compile(r'[\s.(?)!]+')  

with open(conf_dict_com['train_filename'], 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter = '\t')
    for row in reader:
      # if row['language'] == "es":
        test_case.append(row['test_case'])
        source.append(row['source'])
        language.append(row['language'])
        ID.append(row['id'])
        task_1_labels.append(row['task1'])
        task_2_labels.append(row['task2'])

        tweets.append(row['text'].replace("\n", " "))
        simple_cleaning_texts.append(r_white.sub(' ', r_anum.sub('', row['text'].lower())).strip())
        tweetproc_clean_texts.append(tweet_proc.clean(row['text'].replace("\n", " ")))
        se_list = []
        # count = 0
        for se in sent_tokenize(row['text'].replace("\n", " ")):
          se_list.append(se)
        sent_list.append(se_list)

        parse_obj = tweet_proc.parse(row['text'].replace("\n", " "))
        tokenized_tweets.append(tweet_proc.tokenize(row['text'].replace("\n", " ")))
        hashtags.append(make_list(parse_obj.hashtags))
        smileys.append(make_list(parse_obj.smileys))
        emojis.append(make_list(parse_obj.emojis))
        urls.append(make_list(parse_obj.urls))
        mentions.append(make_list(parse_obj.mentions))
        numbers.append(make_list(parse_obj.numbers))
        reserveds.append(make_list(parse_obj.reserved))

for emo_list in emojis:
  texts = []
  for emoji in emo_list:
    for emot in UNICODE_EMO:
      emoji= emoji.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
    # text = emotext(emoji)
      texts.append(emoji.replace("_", " "))
  emoji_texts.append(texts)

for hashset in hashtags:
  segmented_set = []
  for tag in hashset:
    word = tag[1: ]
    # removing the hash symbol
    segmented_set.append(seg_tw.segment(word))
  segmented_hashtags.append(segmented_set)

data_dict = {}
data_dict['task_1_labels'],data_dict['task_2_labels'],data_dict['simple_cleaning_sent_texts'],data_dict['simple_cleaning_texts'], data_dict['test_case'], data_dict['source'], data_dict['language'],data_dict['ID'], data_dict['tweets'], data_dict['tokenized_tweets'], data_dict['hashtags'], data_dict['smileys'], data_dict['emojis'], data_dict['urls'], data_dict['mentions'], data_dict['numbers'], data_dict['reserveds'], data_dict['tweetproc_clean_texts'],data_dict['emoji_texts'], data_dict['segmented_hashtags'] = task_1_labels,task_2_labels,sent_list,simple_cleaning_texts, test_case, source, language, ID, tweets, tokenized_tweets, hashtags, smileys,emojis, urls, mentions, numbers, reserveds, tweetproc_clean_texts,emoji_texts, segmented_hashtags
data_dict['max_num_sent'] = max([len(post_sen) for post_sen in sent_list])
with open(conf_dict_com['data_train'], 'wb') as f:
  pickle.dump(data_dict, f)