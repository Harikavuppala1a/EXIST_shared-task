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
import h5py

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
        

def train(data_dict, conf_dict_com):

    print (conf_dict_com['feat_type'])
    if conf_dict_com['feat_type']== "wordngrams":
        print("Using word based features")
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="word",max_features = conf_dict_com['MAX_FEATURES'],stop_words='english',ngram_range = (1,2))
        bow_transformer_train = count_vec.fit_transform(data_dict['text'][0:data_dict['train_en_ind']])
        bow_transformer_test =count_vec.transform(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test )
        print (train_features.shape)
    elif conf_dict_com['feat_type'] == "charngrams": 
        print("Using char n-grams based features")
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="char",max_features = conf_dict_com['MAX_FEATURES'], ngram_range = (1,5))
        bow_transformer_train = count_vec.fit_transform(data_dict['text'][0:data_dict['train_en_ind']])
        bow_transformer_test =count_vec.transform(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test )         
    elif conf_dict_com['feat_type'] =="glove":
        print("Using glove embeddings")
        emb_size = conf_dict_com['poss_word_feats_emb_dict']['glove']
        emb = get_embeddings_dict(conf_dict_com['feat_type'], emb_size,conf_dict_com["data_folder_name"])
        train_features = get_gove_features(data_dict['text'][0:data_dict['train_en_ind']],emb,emb_size)
        test_features = get_glove_features(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']],emb,emb_size)
    elif conf_dict_com['feat_type'] == "elmo":
        emb_size = conf_dict_com['poss_word_feats_emb_dict']['elmo']
        print("using elmo")
        elmo_filepath = save_folder_name + 'word_vecs~' + conf_dict_com['feat_type'] + '/'
        if not os.path.isfile(elmo_filepath) :
            os.makedirs(elmo_filepath, exist_ok=True)
            elmo = ElmoEmbedder()
            elmo_save_no_pad(data_dict, elmo, elmo_filepath, poss_word_feats_emb_dict[word_feat_name])
        train_features = get_features(data_dict['train_en_ind'],0,elmo_filepath)
        test_features = get_features(len(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']]),data_dict['test_st_ind'],elmo_filepath)
        
    elif conf_dict_com['feat_type'] == "ling":
        ling_filepath = conf_dict_com['save_folder_name'] + 'word_vecs~' + conf_dict_com['feat_type'] + '/'
        if not os.path.isfile(ling_filepath) :
            os.makedirs(ling_filepath, exist_ok=True)
            emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, ling_word_vec_dim = load_ling_word_vec_dicts(conf_dict_com['data_folder_name'])
            ling_word_feat_posts(data_dict['text'], data_dict['max_post_length'], emotion_embed_dict, neut, sentiment_embed_dict, sen_neut, perma_embed_dict, perma_neut, word_embed_dict, word_neut, ling_word_vec_dim, ling_filepath)
        train_features = get_features(data_dict['train_en_ind'],0,ling_filepath)
        test_features = get_features(len(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']]),data_dict['test_st_ind'],ling_filepath)

    elif conf_dict_com['feat_type'].startswith('bert')
        sent_enc_feat_dict ={}
        s_filename = ("%ssent_enc_feat~%s.h5" % (conf_dict_com['save_folder_name'], conf_dict_com['feat_type']))
        if not os.path.isfile(s_filename):
            bert_feats = bert_flat_embed_posts(data_dict['text'], conf_dict_com['poss_word_feats_emb_dict'][conf_dict_com['feat_type']])
            with h5py.File(s_filename, "w") as hf:
                hf.create_dataset('feats', data=bert_feats)
        with h5py.File(s_filename, "r") as hf:
            sent_enc_feat_dict['feats'] = hf['feats'][:data_dict['test_en_ind']]
            train_features = sent_enc_feat_dict['feats'][:data_dict['train_en_ind']]
            test_features = sent_enc_feat_dict['feats'][data_dict['test_st_ind']:data_dict['test_en_ind']]

    return train_features, test_features


start = time.time()
conf_dict_list, conf_dict_com = load_config(sys.argv[1])

os.makedirs(conf_dict_com["output_folder_name"], exist_ok=True)
os.makedirs(conf_dict_com["save_folder_name"], exist_ok=True)

tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_filename"]
if os.path.isfile(tsv_path):
    f_tsv = open(tsv_path, 'a')
else:
    f_tsv = open(tsv_path, 'w')
    f_tsv.write("feature\tmodel\tavg_f\tavg_p\tavg_r\tavg_ac\tstd_f\ttest_mode\n") 

def generate_results(train_feat,test_feat,labels,data_testlabs,num_runs,prob_type,NUM_CLASSES,models,conf_dict_com):
    for model_name in models:
        print (model_name)
        metr_dict = init_metr_dict(prob_type)
        for run_ind in range(num_runs):
            pred, true = classification_model(train_feat, test_feat, labels, data_testlabs, model_name)
            metr_dict = calc_metrics_print(pred, true, metr_dict, NUM_CLASSES, prob_type)
        metr_dict = aggregate_metr(metr_dict, num_runs, prob_type)
        write_results(conf_dict_com['feat_type'],model_name,metr_dict,f_tsv, prob_type,conf_dict_com)

data_dict = load_data(conf_dict_com['filename'], conf_dict_com['data_folder_name'], conf_dict_com['save_folder_name'], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'],conf_dict_com['filename_map'], conf_dict_com['test_mode'])
train_feat, test_feat = train(data_dict,conf_dict_com)
if conf_dict_com['prob_type'] == "binary":
    labels_bin =  trans_labels_bin_classi(data_dict['lab'][:data_dict['train_en_ind']])
    generate_results(train_feat, test_feat,labels_bin[0], data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']],conf_dict_com["num_runs"],data_dict['prob_type'], data_dict['NUM_CLASSES'],conf_dict_com['models'], conf_dict_com)
else:
    labels_mc =trans_labels_mc(data_dict['lab'][:data_dict['train_en_ind']], data_dict['NUM_CLASSES'])
    generate_results(train_feat, test_feat,labels_mc, data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']],conf_dict_com["num_runs"],data_dict['prob_type'], data_dict['NUM_CLASSES'],conf_dict_com['models'], conf_dict_com)   

timeLapsed = int(time.time() - startTime + 0.5)
t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
print(t_str)