import sys
import numpy as np
import os, sys, pickle, csv, sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from evalMeasures import *
from loadPreProc import *
from ling_word_feats import *
from string import punctuation
# from word_embed import *
import nltk
from bert_serving.client import ConcurrentBertClient
from nltk.tokenize import TweetTokenizer
import time
from allennlp.commands.elmo import ElmoEmbedder
import h5py
# from sentence_transformers import SentenceTransformer
import gensim.models as gsm
# from transformers import AutoTokenizer, AutoModelWithLMHead

# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens")
# sent_encoder = SentenceTransformer('xlm-r-100langs-bert-base-nli-mean-tokens')

def get_embedding_weights(filename, sep):
    embed_dict = {}
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(sep)
        embed_dict[row[0]] = row[1:]
    print('Loaded from file: ' + str(filename))
    file.close()
    return embed_dict

def get_embeddings_dict(vector_type, emb_dim,data_folder_name):
    if vector_type == 'sswe':
        emb_dim==50
        sep = '\t'
        vector_file = '../sswe-h.txt'
    elif vector_type =="glove":
        sep = ' '
        vector_file = data_folder_name + 'word_sent_embed/glove.txt'
    
    embed = get_embedding_weights(vector_file, sep)
    
    return embed

def classification_model(X_train, X_test, y_train, y_tested, model_type):
    model = get_model(model_type)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    # print (y_pred)
    # print(y_tested)
    return y_pred, y_tested

def get_model(m_type):
    if m_type == 'logistic_regression':
        logreg = LogisticRegression()
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(n_estimators=conf_dict_com['n_estimators'], n_jobs=-1, class_weight=conf_dict_com['class_weight'])
    elif m_type == "svm":
        logreg = LinearSVC(C=conf_dict_com['c_linear_SVC'],class_weight = conf_dict_com['class_weight'])
    elif m_type == "GBT":
        logreg = GradientBoostingClassifier(n_estimators= conf_dict_com['n_estimators'])
    else:
        print ("ERROR: Please specify a correst model")
        return None
    return logreg

def tf_idf(input_train,input_test,count_vec):
    tfidf_transformer = TfidfTransformer(norm = 'l2')
    bow_transformer_train= count_vec.fit_transform(input_train)
    bow_transformer_test =count_vec.transform(input_test)
    train_features = tfidf_transformer.fit_transform(bow_transformer_train).toarray()
    test_features= tfidf_transformer.transform(bow_transformer_test).toarray()
    return train_features,test_features

def feat_conact(features_word,features_char,features_POS,doc_feat,len_post,adj,text):
    features = []
    for i in range(len(text)): 
            features_text = np.append(features_word[i], features_char[i])
            features_text = np.append(features_text, features_POS[i])
            features_text = np.append(features_text, doc_feat[i])
            features_text = np.append(features_text, [len_post[i], adj[i]])
            features.append(features_text)
    return features

def get_glove_features(Tdata,emb,emb_size):
    features = []
    tknzr = TweetTokenizer()    
    for i in range(len(Tdata)):
            concat = np.zeros(emb_size)
            Tdata[i] = Tdata[i].lower()
            text = ''.join([c for c in Tdata[i] if c not in punctuation])               
            tok = tknzr.tokenize(text)
            toklen = 1
            for wor in range(len(tok)):
                if tok[wor] in emb:
                        toklen += 1
                        flist = [float(i) for i in emb[str(tok[wor])]]
                        concat= flist + concat
            concat = concat/toklen
            features.append(concat)
    return features

def bert_flat_embed_posts(posts, embed_dim):
    posts_arr = np.zeros((len(posts), embed_dim))
    bc = ConcurrentBertClient()
    bert_batch_size = 64
    for ind in range(0, len(posts), bert_batch_size):
        end_ind = min(ind+bert_batch_size, len(posts))
        posts_arr[ind:end_ind, :] = bc.encode(posts[ind:end_ind])
    return posts_arr

def get_features(data, ind, filepath):
    features = []
    for i in range(data):
        arr = np.load(filepath+ str(ind) + '.npy')
        features.append(np.mean(arr, axis=0))
        ind = ind + 1
    return np.asarray(features)

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

def load_features(s_filename):
    sent_enc_feat_dict = {}
    with h5py.File(s_filename, "r") as hf:
        sent_enc_feat_dict['feats'] = hf['feats'][:data_dict['test_en_ind']]
        train_features = sent_enc_feat_dict['feats'][:data_dict['train_en_ind']]
        test_features = sent_enc_feat_dict['feats'][data_dict['test_st_ind']:data_dict['test_en_ind']]
    return train_features,test_features

def save_emojifeats(data_dict,s_filename):
    emoji_feats = np.asarray([getEmojiEmbeddings(i) for i in (data_dict['emojis'])])
    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=emoji_feats)

def save_hashfeats(data_dict,s_filename):
    seg_hashtag =[]
    for hashtag in data_dict['segmented_hashtags']:
        seg_hashtag.append(' '.join(hashtag))
    hash_feat = sent_encoder.encode(seg_hashtag)
    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=hash_feat)

def train(data_dict, conf_dict_com,feat_type):

    print (feat_type)
    if feat_type == "wordngrams":
        print("Using word based features")
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="word",max_features = conf_dict_com['MAX_FEATURES'],stop_words='english',ngram_range = (1,2))
        bow_transformer_train = count_vec.fit_transform(data_dict['text'][0:data_dict['train_en_ind']])
        bow_transformer_test =count_vec.transform(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test )
        print (train_features.shape)
    elif feat_type == "charngrams": 
        print("Using char n-grams based features")
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="char",max_features = conf_dict_com['MAX_FEATURES'], ngram_range = (1,5))
        bow_transformer_train = count_vec.fit_transform(data_dict['text'][0:data_dict['train_en_ind']])
        bow_transformer_test =count_vec.transform(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test )  
        print (train_features.shape)       
    elif feat_type =="glove":
        print("Using glove embeddings")
        emb_size = conf_dict_com['poss_word_feats_emb_dict']['glove']
        emb = get_embeddings_dict(conf_dict_com['feat_type'], emb_size,conf_dict_com["data_folder_name"])
        train_features = get_gove_features(data_dict['text'][0:data_dict['train_en_ind']],emb,emb_size)
        test_features = get_glove_features(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']],emb,emb_size)
    elif feat_type == "elmo":
        emb_size = conf_dict_com['poss_word_feats_emb_dict']['elmo']
        print("using elmo")
        # ad_word_feats['filepath'] = save_fold_path + 'word_vecs~' + word_feat_name + '/' + str(var_tune) + '/'
        elmo_filepath = "saved/word_vecs~elmo/False/"
        #print (elmo_filepath)
        #if not os.path.isfile(conf_dict_com["elmo_filepath"]) :
         #   os.makedirs(elmo_filepath, exist_ok=True)
          #  elmo = ElmoEmbedder()
        #    elmo_save_no_pad(data_dict, elmo, elmo_filepath, poss_word_feats_emb_dict[word_feat_name])
        train_features = get_features(data_dict['train_en_ind'],0,elmo_filepath)
        test_features = get_features(len(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']]),data_dict['test_st_ind'],elmo_filepath)

    return train_features, test_features


start = time.time()
conf_dict_list, conf_dict_com = load_config(sys.argv[1])

os.makedirs(conf_dict_com["output_folder_name"], exist_ok=True)
os.makedirs(conf_dict_com["save_folder_name"], exist_ok=True)

if conf_dict_com['prob_type'] == "multi-class":
   tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_mc_filename"]
else:
    tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_b_filename"]
if os.path.isfile(tsv_path):
    f_tsv = open(tsv_path, 'a')
else:
    f_tsv = open(tsv_path, 'w')
    if conf_dict_com['prob_type'] == "multi-class":
        f_tsv.write("language\tclass_imb\tfeature\tuse_emotions\tuse_hashtags\tuse_empath\tuse_perpective\tuse_hurtlex\trnn_dim\tatt_dim\tavg_f_we\tavg_f_ma\tavg_f_mi\tavg_acc\tavg_p_we\tavg_p_ma\tavg_p_mi\tavg_r_we\tavg_r_ma\tavg_r_mi\tstd_f_we\ttest_mode\tlearnrate\tdrop1\tdrop2\n") 
    else:
        f_tsv.write("language\tclass_imb\tfeature\tuse_emotions\tuse_hashtags\tuse_empath\tuse_perpective\tuse_hurtlex\trnn_dim\tatt_dim\tavg_f\tavg_p\tavg_r\tavg_ac\tstd_f\ttest_mode\tlearnrate\tdrop1\tdrop2\n")  

def generate_results(train_feat,test_feat,labels,data_testlabs,num_runs,prob_type,NUM_CLASSES,models,conf_dict_com):
    for model_name in models:
        print (model_name)
        metr_dict = init_metr_dict(prob_type)
        for run_ind in range(num_runs):
            pred, true = classification_model(train_feat, test_feat, labels, data_testlabs, model_name)
            metr_dict = calc_metrics_print(pred, true, metr_dict, NUM_CLASSES, prob_type)
        metr_dict = aggregate_metr(metr_dict, num_runs, prob_type)
        write_results(conf_dict_com['language'], conf_dict_com['feat_type'],model_name,"False", conf_dict_com['use_emotions'],conf_dict_com['use_hashtags'],conf_dict_com['use_empathfeats'], conf_dict_com['use_perspectivefeats'], conf_dict_com['use_hurtlexfeats'], metr_dict,f_tsv, conf_dict_com['prob_type'],conf_dict_com,"","")

data_dict = load_data(conf_dict_com['filename'], conf_dict_com['data_folder_name'], conf_dict_com['data_train_name'], conf_dict_com['data_test_name'], conf_dict_com['save_folder_name'], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['language'],  conf_dict_com['prob_type'], conf_dict_com['filename_map_list'],conf_dict_com['test_mode'])    

# feat_type_str = ""
train_feat, test_feat = train(data_dict,conf_dict_com,conf_dict_com['feat_type'])
print (train_feat.shape)
# feat_type_str = feat_type_str + conf_dict_com['feat_type_list'][0]
# if len(conf_dict_com['feat_type_list']) >1:
#     for feat_type in conf_dict_com['feat_type_list'][1:]:
#         feat_type_str = feat_type_str + feat_type
#         train_features, test_features = train(data_dict,conf_dict_com,feat_type)
#         train_feat = np.concatenate((train_feat , train_features), axis = 1)
#         test_feat = np.concatenate((test_feat , test_features), axis = 1)
#         print (train_feat.shape)
#         print(test_feat.shape)

if conf_dict_com['prob_type'] == "binary":
    generate_results(train_feat, test_feat,data_dict['task_1_labels'][:data_dict['train_en_ind']], data_dict['task_1_labels'][data_dict['test_st_ind']:data_dict['test_en_ind']],conf_dict_com["num_runs"],conf_dict_com['prob_type'], data_dict['NUM_CLASSES_sd'],conf_dict_com['models'], conf_dict_com)
else:
    generate_results(train_feat, test_feat,data_dict['task_2_labels'][:data_dict['train_en_ind']], data_dict['task_2_labels'][data_dict['test_st_ind']:data_dict['test_en_ind']],conf_dict_com["num_runs"],conf_dict_com['prob_type'], data_dict['NUM_CLASSES_sc'],conf_dict_com['models'], conf_dict_com)   

timeLapsed = int(time.time() - startTime + 0.5)
t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
print(t_str)
