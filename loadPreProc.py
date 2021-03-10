import csv 
from sklearn.utils import shuffle
from ast import literal_eval
import numpy as np   
import os
import re
import pickle
from nltk import sent_tokenize


def load_data(filename, data_path, save_path, test_ratio, valid_ratio, rand_state, max_words_sent, filename_map, test_mode):
  data_dict_filename = ("%sraw_data~%s~%s~%s~%s~%s~%s~%s.pickle" % (save_path, filename[:-4], test_ratio, valid_ratio, rand_state, max_words_sent, filename_map, test_mode))
  if os.path.isfile(data_dict_filename):
    print("loading input data")
    with open(data_dict_filename, 'rb') as f_data:
        data_dict = pickle.load(f_data)
  else:      
    cl_in_filename = ("%sraw_data~%s~%s~%s.pickle" % (save_path, filename[:-4], max_words_sent,filename_map[:-4]))
    if os.path.isfile(cl_in_filename):
      print("loading cleaned unshuffled input")
      with open(cl_in_filename, 'rb') as f_cl_in:
          text, text_sen, label_lists, conf_map = pickle.load(f_cl_in)
    else:
      conf_map = load_map(data_path + filename_map)
      r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
      r_white = re.compile(r'[\s.(?)!]+')
      text = []; label_lists = []; text_sen = []
      with open(data_path + filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter = '\t')
        for row in reader:
          post = str(row['text'])
          row_clean = r_white.sub(' ', r_anum.sub('', post.lower())).strip()
          text.append(row_clean)

          se_list = []
          for se in sent_tokenize(post):
            se_cl = r_white.sub(' ', r_anum.sub('', str(se).lower())).strip()
            if se_cl == "":
              continue
            words = se_cl.split(' ')
            while len(words) > max_words_sent:
              se_list.append(' '.join(words[:max_words_sent]))
              words = words[max_words_sent:]
            se_list.append(' '.join(words))
          text_sen.append(se_list)

          cat_list = str(row['task1']).split(',')
          label_ids = list(set([conf_map['LABEL_MAP'][cat] for cat in cat_list]))
          label_lists.append(label_ids)

      print("saving cleaned unshuffled input")
      with open(cl_in_filename, 'wb') as f_cl_in:
        pickle.dump([text, text_sen, label_lists, conf_map], f_cl_in)

    data_dict = {}  
    data_dict['text'], data_dict['text_sen'], data_dict['lab'] = shuffle(text, text_sen, label_lists, random_state = rand_state)
    train_index = int((1 - test_ratio - valid_ratio)*len(text)+0.5)
    val_index = int((1 - test_ratio)*len(text)+0.5)

    data_dict['max_num_sent'] = max([len(post_sen) for post_sen in data_dict['text_sen'][:val_index]])
    data_dict['max_post_length'] = max([len(post.split(' ')) for post in data_dict['text'][:val_index]])
    data_dict['max_words_sent'] = max_words_sent

    if test_mode:
      data_dict['train_en_ind'] = val_index
      data_dict['test_en_ind'] = len(text)
    else:
      data_dict['train_en_ind'] = train_index
      data_dict['test_en_ind'] = val_index
    data_dict['test_st_ind'] = data_dict['train_en_ind']

    data_dict['FOR_LMAP'] = conf_map['FOR_LMAP']
    data_dict['LABEL_MAP'] = conf_map['LABEL_MAP']
    data_dict['NUM_CLASSES'] = len(data_dict['FOR_LMAP'])
    data_dict['prob_type'] = conf_map['prob_type']

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


def trans_labels_mc(org_lables, NUM_CLASSES):
  label_lists_br = [np.zeros(len(org_lables), dtype=np.int64) for i in range(NUM_CLASSES)]
  for sample_ind, label_ids in enumerate(org_lables):
    for label_id in label_ids:
      label_lists_br[label_id][sample_ind] = 1
  return label_lists_br
  
def trans_labels_bin_classi(org_lables):
  return [np.array([l for l in org_lables], dtype=np.int64)]

