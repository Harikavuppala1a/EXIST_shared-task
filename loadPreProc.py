import csv 
from sklearn.utils import shuffle
from ast import literal_eval
import numpy as np   
import os
import re
import pickle
from nltk import sent_tokenize
from ekphrasis.classes.segmenter import Segmenter
import preprocessor as tweet_proc
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

def strip_list(listie):
  stripped = []
  for item in listie:
    stripped.append(item.strip())
  return stripped

def make_list(proc_obj):
  if proc_obj == None:
    return []
  
  store = []
  for unit in proc_obj:
    store.append(unit.match)
  
  return store


def load_data(filename, data_path, save_path, test_ratio, valid_ratio, rand_state, filename_map_list, lang, prob_type,test_mode):


  # Initializing Lists
  datapoints_count = 0
  see_index = True

  
  tweets = []
  task_1_labels = []
  task_2_labels = []
  ID = []
  source =[]
  language = []
  original_text  = []

  raw_tweet_texts = []
  tokenized_tweets = []
  hashtags = []
  smileys = []
  emojis = []
  urls = []
  mentions = []
  numbers = []
  reserveds = []
  text_case = []
  clean_data = []
  data_dict = {}

  r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
  r_white = re.compile(r'[\s.(?)!]+')  

  seg_tw = Segmenter(corpus = "twitter")
  data_dict_filename = ("%sraw_data~%s~%s~%s~%s~%s~%s~%s~%s.pickle" % (save_path, filename[:-4], test_ratio, valid_ratio, rand_state, filename_map_list, lang, prob_type,test_mode))
  if os.path.isfile(data_dict_filename):
    print("loading input data")
    with open(data_dict_filename, 'rb') as f_data:
        data_dict = pickle.load(f_data)
  else:      
    conf_map = load_map(data_path + filename_map_list[0])
    conf_map1 = load_map(data_path + filename_map_list[1])
    
    with open(data_path + filename, 'r') as csvfile:
      reader = csv.DictReader(csvfile, delimiter = '\t')
      for row in reader:
        datapoints_count += 1
        text_case.append(row['test_case'])
        source.append(row['source'])
        language.append(row['language'])
        task_1_labels.append(conf_map['LABEL_MAP'][str(row['task1'])])
        task_2_labels.append(conf_map1['LABEL_MAP'][str(row['task2'])])
        ID.append(row['id'])
        tweets.append(row['text'].replace("\n", " "))
        row_clean = r_white.sub(' ', r_anum.sub('', row['text'].lower())).strip()
        
        parse_obj = tweet_proc.parse(row['text'].replace("\n", " "))
        tokenized_tweets.append(tweet_proc.tokenize(row['text'].replace("\n", " ")))
        hashtags.append(strip_list(make_list(parse_obj.hashtags)))
        smileys.append(strip_list(make_list(parse_obj.smileys)))
        emojis.append(strip_list(make_list(parse_obj.emojis)))
        urls.append(strip_list(make_list(parse_obj.urls)))
        mentions.append(strip_list(make_list(parse_obj.mentions)))
        numbers.append(strip_list(make_list(parse_obj.numbers)))
        reserveds.append(strip_list(make_list(parse_obj.reserved)))

        raw_tweet_texts.append(tweet_proc.clean(row['text'].replace("\n", " ")))
        clean_data.append(row_clean)
      emoji_texts = []

      for emo_list in emojis:
          texts = []
          for emoji in emo_list:
            for emot in UNICODE_EMO:
              emoji= emoji.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
            # text = emotext(emoji)
              texts.append(emoji.replace("_", " "))
          emoji_texts.append(texts)

      segmented_hashtags = []

      for hashset in hashtags:
          segmented_set = []
          for tag in hashset:
            word = tag[1: ]
            # removing the hash symbol
            segmented_set.append(seg_tw.segment(word))
          segmented_hashtags.append(segmented_set)
            

    data_dict['clean_data'], data_dict['text_case'], data_dict['source'], data_dict['language'],data_dict['task_1_labels'], data_dict['task_2_labels'],data_dict['ID'], data_dict['tweets'], data_dict['tokenized_tweets'], data_dict['hashtags'], data_dict['smileys'], data_dict['emojis'], data_dict['urls'], data_dict['mentions'], data_dict['numbers'], data_dict['reserveds'], data_dict['raw_tweet_texts'],data_dict['emoji_texts'], data_dict['segmented_hashtags'] = shuffle(clean_data, text_case, source, language, task_1_labels, task_2_labels, ID, tweets, tokenized_tweets, hashtags, smileys,emojis, urls, mentions, numbers, reserveds, raw_tweet_texts,emoji_texts, segmented_hashtags, random_state = rand_state)
    train_index = int((1 - test_ratio - valid_ratio)*len(tweets)+0.5)
    val_index = int((1 - test_ratio)*len(tweets)+0.5)

    # data_dict['max_num_sent'] = max([len(post_sen) for post_sen in data_dict['text_sen'][:val_index]])
    data_dict['max_post_length'] = max([len(post.split(' ')) for post in data_dict['tweets'][:val_index]])
    data_dict['max_words_sent'] = 35

    if test_mode:
      data_dict['train_en_ind'] = val_index
      data_dict['test_en_ind'] = len(tweets)
    else:
      data_dict['train_en_ind'] = train_index
      data_dict['test_en_ind'] = val_index
    data_dict['test_st_ind'] = data_dict['train_en_ind']

    data_dict['FOR_LMAP_sd'] = conf_map['FOR_LMAP']
    data_dict['LABEL_MAP_sd'] = conf_map['LABEL_MAP']
    data_dict['NUM_CLASSES_sd'] = len(data_dict['FOR_LMAP_sd'])
    data_dict['prob_type_sd'] = conf_map['prob_type']

    data_dict['FOR_LMAP_sc'] = conf_map1['FOR_LMAP']
    data_dict['LABEL_MAP_sc'] = conf_map1['LABEL_MAP']
    data_dict['NUM_CLASSES_sc'] = len(data_dict['FOR_LMAP_sc'])
    data_dict['prob_type_sc'] = conf_map1['prob_type']

    data_dict['max_num_sent'] = 5

    print("saving input data")
    with open(data_dict_filename, 'wb') as f_data:
      pickle.dump(data_dict, f_data)

  return data_dict

def load_map(filename):
    conf_sep = "----------"
    content = ''
    with open(filename, 'r') as f:
      for line in f:
        line = line.strip()
        if line != '' and line[0] != '#':
          content += line

    items = content.split(conf_sep)
    conf_map = {}
    for item in items:
      parts = [x.strip() for x in item.split('=')]
      conf_map[parts[0]] = literal_eval(parts[1])
    # print(conf_map)
    return conf_map

def load_config(filename):
  print("loading config")
  conf_sep_1 = "----------\n"
  conf_sep_2 = "**********\n"
  conf_dict_list = []
  conf_dict_com = {}
  with open(filename, 'r') as f:
    content = f.read()
  break_ind = content.find(conf_sep_2)  

  nested_comps = content[:break_ind].split(conf_sep_1)
  for comp in nested_comps:
    pairs = comp.split(';')
    conf_dict = {}
    for pair in pairs:
      pair = ''.join(pair.split())
      if pair == "" or pair[0] == '#': 
        continue
      parts = pair.split('=')
      conf_dict[parts[0]] = literal_eval(parts[1])
    conf_dict_list.append(conf_dict)

  lines = content[break_ind+len(conf_sep_2):].split('\n')
  for pair in lines:
    pair = ''.join(pair.split())
    if pair == "" or pair[0] == '#': 
      continue
    parts = pair.split('=')
    conf_dict_com[parts[0]] = literal_eval(parts[1])

  print("config loaded")
  return conf_dict_list, conf_dict_com

def trans_labels_bin_classi(org_lables):
    return [np.array([l for l in org_lables], dtype=np.int64)]
def bin_classi_op_to_label_lists(vec):
  return [[x] for x in vec]

def binary_to_decimal(b_list):
  out1 = 0
  for bit in b_list:
    out1 = (out1 << 1) | bit
  return out1
  
def map_labels_to_num(label_ids, NUM_CLASSES):
  arr = [0] * NUM_CLASSES
  for label_id in label_ids:
    arr[label_id] = 1
  num = binary_to_decimal(arr) 
  return num

def fit_trans_labels_powerset(org_lables, NUM_CLASSES):
  ind = 0
  for_map = {}
  bac_map = {}
  new_labels = np.empty(len(org_lables), dtype=np.int64)
  for s_ind, label_ids in enumerate(org_lables):
    l = map_labels_to_num(label_ids, NUM_CLASSES)
    if l not in for_map:
      for_map[l] = ind
      bac_map[ind] = l
      ind += 1
    new_labels[s_ind] = for_map[l]
  num_lp_classes = ind
  return new_labels, num_lp_classes, bac_map, for_map

def powerset_vec_to_label_lists(vec, bac_map, NUM_CLASSES):
  return [num_to_label_list(bac_map[x], NUM_CLASSES) for x in vec]

def num_to_label_list(num, NUM_CLASSES):
  f_str = ("%sb" % NUM_CLASSES)
  return [ind for ind, x in enumerate(format(num, f_str)) if x == '1']

def trans_labels_multi_classi(org_lables):
  return np.array([l[0] for l in org_lables], dtype=np.int64)

