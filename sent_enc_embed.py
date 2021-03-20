import torch
import os
import sys
import numpy as np
import h5py
sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]),'InferSent'))
# from models import InferSent
import tensorflow as tf
import tensorflow_hub as hub
from bert_serving.client import ConcurrentBertClient
from keras.models import load_model
from bert_features import *
import csv
from bert_modeling import * 

def bert_tune_embed_posts(bert_path,bert_max_seq_len,posts,feat_name,tbert_paths):
	posts_arr = np.zeros((len(posts), 3*bert_max_seq_len))
	print (posts_arr.shape)
	if feat_name == "trainable_tbert" and not os.path.isdir(bert_path):
		build_bert_module(tbert_paths['config_path'],tbert_paths['vocab_path'], tbert_paths['ckpt_path'], bert_path)
	tokenizer = create_tokenizer_from_hub_module(bert_path)
	for ind in range(len(posts)):
		in_examp = InputExample(guid=None,text_a = posts[ind],text_b = None)
		feat = convert_single_example(ind, in_examp, bert_max_seq_len, tokenizer)
		posts_arr[ind] = np.concatenate((np.array(feat.input_ids),np.array(feat.input_mask),np.array(feat.segment_ids)))
	return posts_arr

def bert_flat_embed_posts(posts, embed_dim):
	posts_arr = np.zeros((len(posts), embed_dim))
	bc = ConcurrentBertClient()
	bert_batch_size = 64
	for ind in range(0, len(posts), bert_batch_size):
		end_ind = min(ind+bert_batch_size, len(posts))
		posts_arr[ind:end_ind, :] = bc.encode(posts[ind:end_ind])
	return posts_arr


def sent_enc_featurize(sent_enc_feats_raw, model_type, data_dict, poss_sent_enc_feats_emb_dict, use_saved_sent_enc_feats, save_sent_enc_feats, data_fold_path, save_fold_path, test_mode,bert_path,bert_max_seq_len,tbert_paths,originaltext):
	max_num_attributes = 2
	sent_enc_feats = []
	# var_model_hier = is_model_hier(model_type)
	sent_enc_feat_str = ''
	for sent_enc_feat_raw_dict in sent_enc_feats_raw:
		feat_name = sent_enc_feat_raw_dict['emb']
		sent_enc_feat_str += ("%s~%s~" % (sent_enc_feat_raw_dict['emb'], sent_enc_feat_raw_dict['m_id']))

		sent_enc_feat_dict ={}
		for sent_enc_feat_attr_name, sent_enc_feat_attr_val in sent_enc_feat_raw_dict.items():
			sent_enc_feat_dict[sent_enc_feat_attr_name] = sent_enc_feat_attr_val

		print("computing %s sent feats; test_mode = %s" % (feat_name,test_mode))

		s_filename = ("%ssent_enc_feat~%s~%s.h5" % (save_fold_path, feat_name, originaltext))
		if use_saved_sent_enc_feats and os.path.isfile(s_filename):
			print("loading %s sent feats" % feat_name)
			with h5py.File(s_filename, "r") as hf:
				sent_enc_feat_dict['feats'] = hf['feats'][:data_dict['test_en_ind']]
		else:
			if feat_name.startswith('trainable'):
				feats = bert_tune_embed_posts(bert_path,bert_max_seq_len,data_dict['raw_tweet_texts'],feat_name,tbert_paths)
			elif feat_name == "tbert":
				feats = bert_flat_embed_posts(data_dict['raw_tweet_texts'], poss_sent_enc_feats_emb_dict[feat_name])


			sent_enc_feat_dict['feats'] = feats[:data_dict['test_en_ind']]

			if save_sent_enc_feats:
				print("saving %s sent feats" % feat_name)
				with h5py.File(s_filename, "w") as hf:
					hf.create_dataset('feats', data=feats)

		sent_enc_feats.append(sent_enc_feat_dict)

	sent_enc_feat_str += "~" * ((10 - len(sent_enc_feats_raw)) * max_num_attributes)

	return sent_enc_feats, sent_enc_feat_str[:-1]