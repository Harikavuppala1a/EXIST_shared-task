import os
import re
from nltk import sent_tokenize, word_tokenize
import pickle
import csv

# with open('data/EXIST2021_training.tsv', 'r') as txtfile:
#       with open('exist_posts.txt', 'w') as wfile:
#           reader = csv.DictReader(txtfile, delimiter = '\t')
#           for row in reader:
#               # print (row['post'])
#               if row['task1'] =="sexist":
#                 wfile.write("%s\n" % str(row['text']))

def prepare_data_bert(filename, op_name, data_path,bert_path):
	raw_fname = data_path + filename
	out_name = bert_path + op_name
with open(raw_fname, 'r') as txtfile:
	with open(out_name, 'a') as wfile:
		reader = csv.DictReader(txtfile, delimiter = '\t')
		for row in reader:
            # if row['labels'] != "Non-misygynist" :
                wfile.write("%s\n" % row['post'])
				
# prepare_data_bert('misog_data.txt', 'unlab_bert_predata.txt', 'data/', "../bert_EXIST/tmp/")

# prepare_data_bert('unlab_data_postids.csv', 'unlab_bert_predata.txt', 'data/', "../bert_EXIST/tmp/")

# def bert_pretraining_data(filename, data_path, save_path):
# 	max_seq_len = 1000
# 	raw_fname = data_path + filename
# 	s_name = save_path + filename[:-4] + '_bert_pre.txt'
# 	if os.path.isfile(s_name):
# 		print("already exists")
# 	else:
# 		with open(raw_fname, 'r') as txtfile:
# 			with open(s_name, 'w') as wfile:
# 				post_cnt = 0
# 				for post in txtfile.readlines():
# 					list_sens = []
# 					post_has_big_sens = False
# 					for se in sent_tokenize(post):
# 						if len(word_tokenize(se)) > max_seq_len:
# 							post_has_big_sens = True
# 							break
# 						list_sens.append(se)
# 					if post_has_big_sens:
# 						continue

# 					if post_cnt > 0:
# 						wfile.write("\n")
# 					for se in list_sens:
# 						wfile.write("%s\n" % se)

# 					post_cnt += 1
# 		print("saved %d bert pretraining data" % post_cnt)
# bert_pretraining_data('unlab_sans_test.txt', "data/", "../bert/tmp/")

# screen -L python create_pretraining_data.py \
#   --input_file=tmp/unlab_bert_predata_sc_miso.txt \
#   --output_file=tmp/tf_examples.tfrecord \
#   --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=120 \
#   --max_predictions_per_seq=18 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5

# screen -L python run_pretraining.py \
#   --input_file=tmp/tf_examples.tfrecord \
#   --output_dir=tmp/pretraining_output \
#   --do_train=True \
#   --do_eval=True \
#   --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
#   --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt \
#   --train_batch_size=25 \
#   --max_seq_length=120 \
#   --max_predictions_per_seq=18 \
#   --num_train_steps=100000 \
#   --num_warmup_steps=10000 \
#   --learning_rate=2e-5