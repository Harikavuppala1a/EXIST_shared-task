
import random
import pickle
import re
import time
import datetime
import sys
import statistics

import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModel

import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel, RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import gensim.models as gsm

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from empath import Empath

from analysis import *
from features import *
from load_data import *

from tqdm import tqdm 
import gc
import os
import csv
import re,requests,json

sys.setrecursionlimit(10000)
conf_dict_list, conf_dict_com = load_config(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] =conf_dict_com['GPU_ID']

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print (device)
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    print (device)

class EXISTDataset(Dataset):
  """ Data loader to load the data for the Torch """
  def __init__(self, dataPath, isDF = False):
    if isDF:
      self.df = pd.DataFrame.from_dict(dataPath)
    else:
      data = pickle.load(open(dataPath,'rb'))
      self.df = pd.DataFrame.from_dict(data)
    # print (self.df)
  def __len__(self):
    return len(self.df)
  def __getitem__(self,index):
    return self.df.iloc[index]


def set_seed(seed):
     # """ Sets all seed to the given value, so we can reproduce (:3) """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

tokenizer = AutoTokenizer.from_pretrained(conf_dict_com['pre-trained_filename'])
tmodel = AutoModel.from_pretrained(conf_dict_com['pre-trained_filename'])

conf_map_sc = load_map(conf_dict_com['filename_map_sc'])
conf_map_sd = load_map(conf_dict_com['filename_map_sd'])

class ExampleFeautres(object):
    """ Contains the dataset in a batch friendly feaute set """
    def __init__(self, id,task_1_labels, task_2_labels, input_ids, input_mask,input_length, emoji_texts,  segmented_hashtags, empath_feats, perspective_feat, hurtlex_feat):
      self.id  = id
      self.task_1 = task_1_labels
      self.task_2 = task_2_labels
      self.input_ids = input_ids
      self.input_mask = input_mask
      self.input_length = input_length
      self.emoji = torch.tensor(emoji_texts)
      self.hash = torch.tensor(segmented_hashtags)
      self.empath = torch.tensor(empath_feats)
      self.perspective = torch.tensor(perspective_feat)
      self.hurtlex = torch.tensor(hurtlex_feat)

class Example(object):
  """ Contains the data for one example from the dataset """
  def __init__(self, id, task_1_labels, task_2_labels, ID, simple_cleaning_sent_texts, simple_cleaning_texts, emoji_texts,  segmented_hashtags, language,max_num_sent):
    self.id  = id
    self.task_1 = task_1_labels
    self.task_2 = task_2_labels
    self.exist_id = ID
    self.full_sent_tweet =  simple_cleaning_sent_texts
    self.tweet_raw_text = simple_cleaning_texts
    self.emoji = emoji_texts
    self.segmented_hash = segmented_hashtags
    self.lang = language
    self.max_sent_cnt =max_num_sent

lexicon = Empath()
lexicon_en = read_lexicon('../hurtlex_en.tsv',len(conf_dict_com['hurtlex_feat_list']),conf_dict_com['hurtlex_feat_list'])
# lexicon_es = read_lexicon('hurtlex_es.tsv', len(conf_dict_com['hurtlex_feat_list']),conf_dict_com['hurtlex_feat_list'])

def convertExamplesToFeature(example):
  """ Given a data row convert it to feautres so it's batch friendly """
  raw_text = example.tweet_raw_text 
  sent_text = example.full_sent_tweet
  tokens = tokenizer.tokenize(raw_text)
  if (len(tokens) > (conf_dict_com['max_seq_length']-2)):
    tokens = tokens[: (conf_dict_com['max_seq_length']-2)]
  tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)
  input_length = len(input_ids)
  padding = [0] * (conf_dict_com['max_seq_length'] - len(input_ids))
  input_ids += padding
  input_mask += padding

  hashtags = ' '.join(example.segmented_hash)
  hash_ids= torch.tensor(tokenizer.encode(hashtags)).unsqueeze(0)  # Batch size 1
  outputs = tmodel(hash_ids)
  hashembs = outputs[1]

  emojiVec = getEmojiEmbeddings(example.emoji,conf_dict_com['emoji_filename'])

  # empath_feats = np.zeros(len(empath_feats_list))
  empath_feats = np.zeros((conf_dict_com['max_sent_cnt'], len(conf_dict_com['empath_feats_list'])))
  l = min(conf_dict_com['max_sent_cnt'],len(sent_text))
  for in_ind,sent in enumerate(sent_text[:l]):
    # print (sent)
    empath_dict = lexicon.analyze(sent,categories = conf_dict_com['empath_feats_list'],normalize=True)
    # print (empath_dict)
    for di_ind, tup in enumerate(empath_dict.items()):
      k,v = tup
      empath_feats[in_ind,di_ind] = v

  perspective_feat = save_perspectivefeats(sent_text,example.lang,conf_dict_com['max_sent_cnt'])

  hurtlex_feat = np.zeros((conf_dict_com['max_sent_cnt'],2*len(conf_dict_com['hurtlex_feat_list'])))
  l = min(conf_dict_com['max_sent_cnt'],len(sent_text))
  if example.lang == "en":
      for in_ind,sent in enumerate(sent_text[:l]):
        hurtlex_feat[in_ind,:] =  check_presence(sent.lower(), lexicon_en, len(conf_dict_com['hurtlex_feat_list']))
  else:
      for in_ind,sent in enumerate(sent_text[:l]):
        hurtlex_feat[in_ind,:] =  check_presence(sent.lower(), lexicon_es, len(conf_dict_com['hurtlex_feat_list']))

  task1 = conf_map_sd['LABEL_MAP'][example.task_1]
  task2 = conf_map_sc['LABEL_MAP'][example.task_2]
  id = int(example.exist_id)
  return ExampleFeautres(id, task1, task2, input_ids, input_mask, input_length, emojiVec, hashembs.reshape(-1), empath_feats, perspective_feat, hurtlex_feat)

def getDataset(input_features):
    """
    Mappings for index-> features 
    0 -> ID
    1 -> input ids
    2 -> input masks
    3 -> input lengths 
    4 -> hash embs 
    5 -> emoji embs 
    6 -> empath embs 
    7 -> perspective embs
    8 -> hurtlex embs 
    9 -> task1
    10 -> task2
    """
    all_input_page_ids = torch.tensor([f.id for f in input_features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in input_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in input_features], dtype=torch.long)
    all_input_lengths = torch.tensor([f.input_length for f in input_features], dtype=torch.long)
    all_hash_embs = torch.stack([f.hash for f in input_features])
    all_emoji_embs = torch.stack([f.emoji for f in input_features])
    all_empath_embs = torch.stack([f.empath for f in input_features])
    all_perspective_embs = torch.stack([f.perspective for f in input_features])
    all_hurtlex_embs = torch.stack([f.hurtlex for f in input_features])
    all_task_1 = torch.tensor([f.task_1 for f in input_features], dtype=torch.long)
    all_task_2 = torch.tensor([f.task_2 for f in input_features], dtype=torch.long)

    dataset = TensorDataset(all_input_page_ids.to(device), all_input_ids.to(device), all_input_mask.to(device),all_input_lengths.to(device), all_hash_embs.to(device), all_emoji_embs.to(device),all_empath_embs.to(device),all_perspective_embs.to(device),all_hurtlex_embs.to(device), all_task_1.to(device),  all_task_2.to(device))
    return dataset

def train_val_dataset(dataset):
    train_idx = list(range(0, len(dataset)))
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    return datasets,train_idx

def getDataloader(path_to_pickle, batch_size,data):
  tempDataset = EXISTDataset(path_to_pickle)
  # print (tempDataset)
  input_features = []
  if data == "train":
    filename =  "../allfeatures_sent_translated_train.pickle"
  else:
    filename = "../allfeatures_sent_translated_test.pickle"
  if os.path.isfile(filename):
    input_features = pickle.load(open(filename,'rb'))
  else:   
    for i in tqdm(range(len(tempDataset))):
      example = Example(i,tempDataset[i]['task_1_labels'],tempDataset[i]['task_2_labels'],tempDataset[i]['ID'], tempDataset[i]['simple_cleaning_sent_texts'],tempDataset[i]['simple_cleaning_texts'], tempDataset[i]['emoji_texts'],tempDataset[i]['segmented_hashtags'], tempDataset[i]['language'], tempDataset[0]['max_num_sent'])
      input_feature = convertExamplesToFeature(example)
      input_features.append(input_feature)
    with open(filename, 'wb') as f_cl_in:
        pickle.dump(input_features, f_cl_in)
  dataset = getDataset(input_features)
  set_seed(42)
  dd,traindata= train_val_dataset(dataset)
  if data == "train":
    dataloader = DataLoader(dd['train'], sampler = RandomSampler(dd['train']), batch_size=batch_size, drop_last=True)
  else:
    dataloader = DataLoader(dd['train'] , batch_size=batch_size, drop_last=True)
  return dataloader 

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class ClassificationHead(nn.Module):
  """ Classification head for the Roberta Model """ 
  # def __init__(self, numberOfClasses, hidden_size_bert, hidden_size_post_feats, dropout_val = 0.2):
  def __init__(self, numberOfClasses, hidden_size_att,hidden_size_post_feats,dropout_val):
    super().__init__()
    # self.denseInit = nn.Linear(hidden_size_post_feats, hidden_size_bert)
    # print (hidden_size_att)
    self.denseInit = nn.Linear(hidden_size_post_feats,hidden_size_att)
    # self.dense = nn.Linear(hidden_size_att, hidden_size_att)
    self.dropout = nn.Dropout(dropout_val)
    self.output = nn.Linear(hidden_size_att, numberOfClasses)
  def forward(self, x):
    # print(x.shape)
    x = self.dropout(x)
    x = self.denseInit(x)
    x = torch.tanh(x)
    x = self.dropout(x)
    # x = self.dense(x)
    # x  = torch.tanh(x)
    # x = self.dropout(x)
    x  = self.output(x)
    return x

class TextClassification(nn.Module):
  """ Classifier with feature injection """
  def __init__(self, numberOfClasses):
     super(TextClassification, self).__init__()
     self.tmodel = AutoModel.from_pretrained(conf_dict_com['pre-trained_filename'])
     self.lstm = nn.LSTM(self.tmodel.config.hidden_size,conf_dict_com['rnn_dim'],num_layers=1,bidirectional=True,batch_first=True)
     self.lstm_p = nn.LSTM(conf_dict_com['perspective_len'],conf_dict_com['rnn_dim'],num_layers=1,bidirectional=True,batch_first=True)
     self.lstm_h = nn.LSTM(2*len(conf_dict_com['hurtlex_feat_list']),conf_dict_com['rnn_dim'],num_layers=1,bidirectional=True,batch_first=True)
     self.lstm_e = nn.LSTM(len(conf_dict_com['empath_feats_list']),conf_dict_com['rnn_dim'],num_layers=1,bidirectional=True,batch_first=True)
     self.hidden_cell = (torch.zeros(2, conf_dict_com['batch_size'],conf_dict_com['rnn_dim']),
                            torch.zeros(2, conf_dict_com['batch_size'],conf_dict_com['rnn_dim']))
     self.attention_layer = Attention(conf_dict_com['att_dim'], conf_dict_com['max_seq_length'])
     self.attention_layer_feat = Attention(conf_dict_com['att_dim'], conf_dict_com['max_sent_cnt'])
     self.classifier = ClassificationHead(numberOfClasses, conf_dict_com['att_dim'],conf_dict_com['att_dim']*3+300, conf_dict_com['dropout_val'])

  def forward(self, input_seq, attention_mask,perspective,empath,hurtlex,emoji,hashtag):
    seq_output_t = self.tmodel(input_seq, attention_mask=attention_mask)[0]
    h, c = self.hidden_cell
    lstm_output_t, _  = self.lstm(seq_output_t,(h.to(device), c.to(device)))
    att_output = self.attention_layer(lstm_output_t)

    lstm_output_p, _  = self.lstm_p(perspective,(h.to(device), c.to(device)))
    lstm_output_h, _  = self.lstm_h(hurtlex,(h.to(device), c.to(device)))
    lstm_output_e, _  = self.lstm_e(empath,(h.to(device), c.to(device)))
    att_output_p = self.attention_layer_feat(lstm_output_p)
    att_output_h = self.attention_layer_feat(lstm_output_h)
    att_output_e = self.attention_layer_feat(lstm_output_e)
    concat_output = torch.cat([att_output,att_output_h,att_output_e,emoji], axis = 1)
    output = self.classifier(concat_output)
    return output


def modelEvaluate(model, valid_dataloader, task , evaluate):
  gc.collect()
  if conf_dict_com['task'] == 1:
    taskIndex = 9
  else:
    taskIndex = 10
  if evaluate != "train":
    model.eval()
  predictions, true_labels = [], []
  logits = []
  # Predict 
  for batch in valid_dataloader:
    # Add batch to GPU
    b_input_ids = batch[1]
    b_input_mask = batch[2]
    b_labels = batch[taskIndex]
    b_emoji = batch[5]
    b_hashtag = batch[4]
    b_empath = batch[6]
    b_perspective = batch[7]
    b_hurtlex = batch[8]
    with torch.no_grad():
      pred = model(b_input_ids,b_input_mask,b_perspective.float(),b_empath.float(),b_hurtlex.float(),b_emoji.float(),b_hashtag.float())
    logits.append(pred.detach().cpu().numpy())
    label_ids = b_labels.to('cpu').numpy()
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)
  flat_true_labels = np.concatenate(true_labels, axis = 0)
  predictions = []
  for i in logits:
    for j in i:
      predictions.append(j)
  flat_predictions = [np.argmax(i) for i in predictions]
  assert(len(flat_predictions) == len(flat_true_labels))
  return flat_predictions, flat_true_labels

def make_optim(model, rate = 2e-5):
  return AdamW(model.parameters(),
                lr = rate, # default = 5e-5, using 2e-5
                eps = 1e-8) # default = 1e-8

def train_model(train_dataloader, valid_dataloader, task, path):
  """ Train Loop for the model """
  scale = 1
  if conf_dict_com['att_dim'] == 2:
    classNum = 6
    taskIndex = 10
  else:
    classNum = 2
    taskIndex = 9

  total_steps = len(train_dataloader)
  print("Start")

  model = TextClassification(classNum) # task 1
  model.cuda() 
 
  loss_function = nn.CrossEntropyLoss()
  epoch_loss = 0
  batch_accuracy_scores = []
  global_pred = []
  global_label = []

  present_rate = 2e-5
  old_best = -1
  epoch = 0

  while(1):
    # when the learn rate falls below a lower threshold, you stop your training
    # until that moment, march on
    epoch += 1
    print("\nEpoch:", epoch)
    print("Present Rate: " + str(present_rate))
    optimizer = make_optim(model, present_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps = 0, # Default value in run_glue.py
                                              num_training_steps = total_steps)
    gc.collect()
    model.train()
    epoch_loss = 0
    batch_accuracy_scores = []
    train_data_count = float(len(train_dataloader))

    # to check if performance with default weights
    predictions, true_labels = modelEvaluate(model, train_dataloader, task,"train")
    if conf_dict_com['task'] == 1:
      score_f1 = f1_score(true_labels, predictions)
    else:
      score_f1 = f1_score(true_labels, predictions, average = 'macro')
    score_acc = accuracy_score(true_labels, predictions)
    print("Validation Macro: " + str(score_f1))
    print ("ACcuracy:" + str(score_acc))

    if conf_dict_com['task'] == 1:
      score = score_acc
    else:
      score = score_f1

    if (score > old_best):
      print("Continuing on track")
      old_best = score

      # delete previous best 
      delete_filename = path
      open(delete_filename, 'w').close() # overwrite and make the file blank instead
      os.remove(delete_filename) # delete the blank file from google drive will move the file to bin instead
      torch.save(model.state_dict(), path)

    else:
      print("Backtrack")
      model.load_state_dict(torch.load(path))
      present_rate /= (4 * scale)
      scale *= 4
      if present_rate < 1e-8:
        break

    for i, batch in enumerate(train_dataloader):

        b_input_ids = batch[1]
        b_input_mask = batch[2]
        b_labels = batch[taskIndex]
        b_emoji = batch[5]
        b_hashtag = batch[4]
        b_empath = batch[6]
        b_perspective = batch[7]
        b_hurtlex = batch[8]
        pred = model(b_input_ids,b_input_mask,b_perspective.float(),b_empath.float(),b_hurtlex.float(),b_emoji.float(),b_hashtag.float())

        loss = loss_function(pred.view(-1, classNum), b_labels.view(-1))
        with torch.no_grad():
          epoch_loss += (loss.item() * len(b_labels))
          global_pred.append(pred)
          global_label.append(b_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  return model


train_dataloader = getDataloader(conf_dict_com["data_train"] ,conf_dict_com["batch_size"],"train")
valid_dataloader = getDataloader(conf_dict_com["data_test"] ,conf_dict_com["batch_size"],"test")

def loadModel(model_path, task = 1):
  """ Code to load a model based on the saved points """
  if conf_dict_com['task'] == 1:
    classNum = 2
  else:
    classNum = 6

  model = TextClassification(classNum)
  model.load_state_dict(torch.load(model_path))
  return model

f1 = []
fw =[]
acc =[]
prec =[]
recall =[]
true_labs =[]
pred_labs =[]
for i in range(conf_dict_com['num_runs']):
  gc.collect()
  path = conf_dict_com['model_name'] + str(i) + ".pt"
  if os.path.isfile(path):
    model = loadModel(path, conf_dict_com['task'])
    model.cuda()
  else:
    model = train_model(train_dataloader, valid_dataloader, conf_dict_com['task'],path)

  predictions, true_labels = modelEvaluate(model, valid_dataloader, conf_dict_com['task'], "test")
  if conf_dict_com['task'] ==1:
    f1.append(f1_score(true_labels, predictions))
    prec.append(precision_score(true_labels, predictions))
    recall.append(recall_score(true_labels, predictions))
  else:
    f1.append(f1_score(true_labels, predictions, average = 'macro'))
    fw.append(f1_score(true_labels, predictions, average = 'weighted'))
    prec.append(precision_score(true_labels, predictions, average = 'macro'))
    recall.append(recall_score(true_labels, predictions, average = 'macro'))
  acc.append(accuracy_score(true_labels,predictions))
  true_labs.append(true_labels)
  pred_labs.append(predictions)
  if i ==0:
    ids = []
    for batch in valid_dataloader:
      ids.append(batch[0])
  
  # print ("f1score: %.3f" % f1_score(true_labels, predictions))
  # print ("Accuracy: %.3f" % accuracy_score(true_labels,predictions))
  del model
  # model.cpu()
if os.path.isfile(conf_dict_com['tsv_path']):
    f_tsv = open(conf_dict_com['tsv_path'], 'a')
else:
    f_tsv = open(conf_dict_com['tsv_path'], 'w')
    if conf_dict_com['task'] == 1:
      f_tsv.write("task\tmodel_name\tfeatures\tnonlnearity\tloss_function\t dropout\trnndim\tattdim\temoji \thashtag \tempath \t perspective\t hurtlex\tstd_f1\tstd_acc\truns_f1\truns_accuracy\truns_precision\truns_recall\n") 
    else:
      f_tsv.write("task\tmodel_name\tfeatures\tnonlnearity\tloss_function\t dropout\trnndim\tattdim\temoji \thashtag \tempath \t perspective\t hurtlex\tstd_f1\tstd_acc\truns_f1\truns_fw\truns_accuracy\truns_precision\truns_recall\n") 

std_f1 = statistics.stdev(f1)
std_acc = statistics.stdev(acc)
if conf_dict_com['task'] == 1:
  f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (conf_dict_com['task'],conf_dict_com['model_name'],conf_dict_com['features'],conf_dict_com['non_linearity'], conf_dict_com['loss_function'], str(conf_dict_com['dropout_val']),conf_dict_com['rnn_dim'],conf_dict_com['att_dim'],conf_dict_com['emoji'],conf_dict_com['hashtag'],conf_dict_com['empath'],conf_dict_com['perspective'],conf_dict_com['hurtlex'],std_f1,std_acc,statistics.mean(f1),statistics.mean(acc), statistics.mean(prec), statistics.mean(recall)))
else:
  f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (conf_dict_com['task'],conf_dict_com['model_name'],conf_dict_com['features'],conf_dict_com['non_linearity'], conf_dict_com['loss_function'], str(conf_dict_com['dropout_val']),conf_dict_com['rnn_dim'],conf_dict_com['att_dim'],conf_dict_com['emoji'],conf_dict_com['hashtag'],conf_dict_com['empath'],conf_dict_com['perspective'],conf_dict_com['hurtlex'],std_f1,std_acc,statistics.mean(f1),statistics.mean(f1),statistics.mean(acc), statistics.mean(prec), statistics.mean(recall)))

if conf_dict_com['analysis']:
  if conf_dict_com['task'] == 1:
    classNum = 2
    conf_map = conf_map_sd
  else:
    classNum = 6
    conf_map = conf_map_sc
  insights_results_lab(pred_labs, true_labs,conf_dict_com['num_runs'],classNum,conf_dict_com['model_name'],conf_dict_com['task'],conf_map)
  testdata_analysis(pred_labs[0],true_labs[0],ids, conf_dict_com['test_filename'])