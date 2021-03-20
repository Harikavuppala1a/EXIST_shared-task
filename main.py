import os
import time
import sys
from sent_enc_embed import sent_enc_featurize
from word_embed import word_featurize
from neural_approaches import *
from bert_serving.client import ConcurrentBertClient
import h5py
import gensim.models as gsm

# from TraditionalML_LP import save_emojifeats, save_hashfeats

sys.setrecursionlimit(10000)
conf_dict_list, conf_dict_com = load_config(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = conf_dict_com['GPU_ID']
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

conf_dict_list, conf_dict_com = load_config(sys.argv[1])

os.makedirs(conf_dict_com["output_folder_name"], exist_ok=True)
os.makedirs(conf_dict_com["save_folder_name"], exist_ok=True)

if conf_dict_com['prob_type'] == "multi-class":
   tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_mc_filename"]
else:
    tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_b_filename"]
data_dict = load_data(conf_dict_com['filename'], conf_dict_com['data_folder_name'], conf_dict_com['save_folder_name'], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'],conf_dict_com['filename_map_list'], conf_dict_com['language'],  conf_dict_com['prob_type'] ,conf_dict_com['test_mode'])    

if os.path.isfile(tsv_path):
    f_tsv = open(tsv_path, 'a')
else:
    f_tsv = open(tsv_path, 'w')
    if conf_dict_com['prob_type'] == "multi-class":
        f_tsv.write("language\toriginaltext\tclass_imb\tfeature\tmodel\tuse_emotions\tuse_hashtags\trnn_dim\tatt_dim\tavg_f_we\tavg_f_ma\tavg_f_mi\tavg_acc\tavg_p_we\tavg_p_ma\tavg_p_mi\tavg_r_we\tavg_r_ma\tavg_r_mi\tstd_f_we\ttest_mode\tlearnrate\tdrop1\tdrop2\n") 
    else:
        f_tsv.write("language\toriginaltext\tclass_imb\tfeature\tmodel\tuse_emotions\tuse_hashtags\trnn_dim\tatt_dim\tavg_f\tavg_p\tavg_r\tavg_ac\tstd_f\ttest_mode\tlearnrate\tdrop1\tdrop2\n") 

if conf_dict_com['prob_type'] == "binary":
    data_dict['NUM_CLASSES'] = data_dict['NUM_CLASSES_sd']
    data_dict['FOR_LMAP'] = data_dict['FOR_LMAP_sd']
    data_dict['LABEL_MAP'] = data_dict['LABEL_MAP_sd']
    data_dict['NUM_CLASSES'] =data_dict['NUM_CLASSES_sd']
    data_dict['prob_type'] = data_dict['prob_type_sd']
    data_dict['lab'] = data_dict['task_1_labels']
else:
    data_dict['NUM_CLASSES'] = data_dict['NUM_CLASSES_sc']
    data_dict['FOR_LMAP'] = data_dict['FOR_LMAP_sc']
    data_dict['LABEL_MAP'] = data_dict['LABEL_MAP_sc']
    data_dict['NUM_CLASSES'] =data_dict['NUM_CLASSES_sc']
    data_dict['prob_type'] = data_dict['prob_type_sc']
    data_dict['lab'] = data_dict['task_2_labels']

print (data_dict['raw_tweet_texts'][1])
if conf_dict_com['original_text']:
    data_dict['raw_tweet_texts'] = data_dict['tweets']
    print (data_dict['raw_tweet_texts'][1])
if conf_dict_com['simple_cleaning']:
    data_dict['raw_tweet_texts'] = data_dict['clean_data']

if conf_dict_com['use_emotions']:
    s_filename = ("%semoji_enc_feat~%s.h5" % (conf_dict_com['save_folder_name'], conf_dict_com['language']))
    if not os.path.isfile(s_filename):
        save_emojifeats(data_dict,s_filename)
    with h5py.File(s_filename, "r") as hf:
        emoji_array = hf['feats'][:data_dict['test_en_ind']]
else:
    emoji_array = []

if conf_dict_com['use_hashtags']:
    s_filename = ("%shashtags_enc_feat~%s.h5" % (conf_dict_com['save_folder_name'],conf_dict_com['language']))
    if not os.path.isfile(s_filename):
        save_hashfeats(data_dict,s_filename)
    with h5py.File(s_filename, "r") as hf:
            hashtag_array = hf['feats'][:data_dict['test_en_ind']]
else:
    hashtag_array =[]

print (data_dict['NUM_CLASSES'])
metr_dict = init_metr_dict(conf_dict_com['prob_type'])
feat_type_str = ""
classi_probs_label_str = "None"
for conf_dict in conf_dict_list:
    for prob_trans_type in conf_dict["prob_trans_types"]:
        trainY_list, trainY_noncat_list, num_classes_list, bac_map = transform_labels(data_dict['lab'][:data_dict['train_en_ind']], prob_trans_type, conf_dict_com['test_mode'], conf_dict_com["save_folder_name"], data_dict['NUM_CLASSES'], data_dict['prob_type'], conf_dict_com['classi_probs_label_info'], classi_probs_label_str, conf_dict_com['use_saved_data_stuff'], conf_dict_com['save_data_stuff'])
        for class_imb_flag in conf_dict["class_imb_flags"]:
            loss_func_list, nonlin, out_vec_size_list, cw_list = class_imb_loss_nonlin(trainY_noncat_list, class_imb_flag, num_classes_list, conf_dict_com['prob_type'], conf_dict_com['test_mode'], conf_dict_com["save_folder_name"], conf_dict_com['use_saved_data_stuff'], conf_dict_com['save_data_stuff'])
            for model_type in conf_dict["model_types"]:
                for word_feats_raw in conf_dict["word_feats_l"]:
                    word_feats, word_feat_str = word_featurize(word_feats_raw, model_type, data_dict, conf_dict_com['poss_word_feats_emb_dict'], conf_dict_com['use_saved_word_feats'], conf_dict_com['save_word_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"])
                    for sent_enc_feats_raw in conf_dict["sent_enc_feats_l"]:
                        sent_enc_feats, sent_enc_feat_str = sent_enc_featurize(sent_enc_feats_raw, model_type, data_dict, conf_dict_com['poss_sent_enc_feats_emb_dict'], conf_dict_com['use_saved_sent_enc_feats'], conf_dict_com['save_sent_enc_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"],conf_dict_com['bert_path'],conf_dict_com['bert_max_seq_len'],conf_dict_com["tbert_paths"], conf_dict_com['original_text'])
                        for num_cnn_filters in conf_dict["num_cnn_filters"]:
                            for max_pool_k_val in conf_dict["max_pool_k_vals"]:
                                for cnn_kernel_set in conf_dict["cnn_kernel_sets"]:
                                    cnn_kernel_set_str = str(cnn_kernel_set)[1:-1].replace(', ','_')
                                    for rnn_type in conf_dict["rnn_types"]:
                                        for rnn_dim in conf_dict["rnn_dims"]:
                                            for att_dim in conf_dict["att_dims"]:
                                                for stack_rnn_flag in conf_dict["stack_rnn_flags"]:
                                                    mod_op_list_save_list = []
                                                    for thresh in conf_dict["threshes"]:
                                                        startTime = time.time()
                                                        info_str = "model: %s, word_feats = %s, sent_enc_feats = %s, classi_probs_label_info = %s, prob_trans_type = %s, class_imb_flag = %s, num_cnn_filters = %s, cnn_kernel_set = %s, rnn_type = %s, rnn_dim = %s, att_dim = %s, max_pool_k_val = %s, stack_rnn_flag = %s, thresh = %s, test mode = %s" % (model_type,word_feat_str,sent_enc_feat_str,classi_probs_label_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set,rnn_type,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag, thresh, conf_dict_com["test_mode"])
                                                        fname_part = ("%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s" % (model_type,word_feat_str,sent_enc_feat_str,classi_probs_label_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,rnn_type,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag, conf_dict_com['use_emotions'], conf_dict_com['use_hashtags'], conf_dict_com['original_text'], conf_dict_com["test_mode"]))
                                                        pred_vals_across_runs = []
                                                        for run_ind in range(conf_dict_com["num_runs"]):
                                                            print('run: %s; %s\n' % (run_ind, info_str))
                                                            if run_ind < len(mod_op_list_save_list):
                                                                mod_op_list = mod_op_list_save_list[run_ind]   
                                                            else:
                                                                mod_op_list = []
                                                                # print (loss_func_list)
                                                                # print (cw_list)
                                                                # print (out_vec_size_list)
                                                                # print (trainY_list)
                                                                for m_ind, (loss_func, cw, out_vec_size, trainY) in enumerate(zip(loss_func_list, cw_list, out_vec_size_list, trainY_list)):
                                                                    # print ("in loop")
                                                                    mod_op, att_op = train_predict(word_feats, sent_enc_feats, trainY, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, cw, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val,stack_rnn_flag,cnn_kernel_set, m_ind, run_ind, conf_dict_com["save_folder_name"], conf_dict_com["use_saved_model"], conf_dict_com["gen_att"], conf_dict_com["LEARN_RATE"], conf_dict_com["dropO1"], conf_dict_com["dropO2"], conf_dict_com["BATCH_SIZE"], conf_dict_com["EPOCHS"], conf_dict_com["save_model"], conf_dict_com["save_trained_mod"],conf_dict_com['n_fine_tune_layers'],conf_dict_com['bert_path'],conf_dict_com['output_representation'],conf_dict_com['bert_trainable'], conf_dict_com['use_emotions'],emoji_array, conf_dict_com['use_hashtags'], hashtag_array)
                                                                    mod_op_list.append((mod_op, att_op))
                                                                mod_op_list_save_list.append(mod_op_list) 
                                                            pred_vals, true_vals, metr_dict = evaluate_model(mod_op_list, data_dict, bac_map, prob_trans_type, metr_dict, thresh, conf_dict_com['gen_att'], conf_dict_com["output_folder_name"], ("%s~%d" % (fname_part,run_ind)), conf_dict_com['classi_probs_label_info'])
                                                            pred_vals_across_runs.append(pred_vals)
                                                        # f_res.write("%s\n\n" % info_str)
                                                        print("%s\n" % info_str)
                                                        # print(metr_dict)
                                                        metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs"], data_dict['prob_type'])
                                                        # write_results(metr_dict, f_res, f_tsv, data_dict['prob_type'], model_type,word_feat_str,sent_enc_feat_str,classi_probs_label_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,thresh,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag,rnn_type,conf_dict_com)
                                                        write_results(conf_dict_com['language'], conf_dict_com['original_text'], class_imb_flag, sent_enc_feat_str,conf_dict_com['model'],conf_dict_com['use_emotions'],conf_dict_com['use_hashtags'], metr_dict,f_tsv, conf_dict_com['prob_type'],conf_dict_com,rnn_dim,att_dim)
                                                        if conf_dict_com['gen_insights']:                                                  
                                                            insights_results_lab(pred_vals_across_runs, true_vals, data_dict['lab'][0:data_dict['train_en_ind']], fname_part, conf_dict_com["output_folder_name"],conf_dict_com["num_runs"],data_dict['NUM_CLASSES'],data_dict['FOR_LMAP'] )
                                                            insights_results(pred_vals_across_runs[0], true_vals, data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']], data_dict['text_sen'][data_dict['test_st_ind']:data_dict['test_en_ind']], fname_part, conf_dict_com["output_folder_name"])
                                                        timeLapsed = int(time.time() - startTime + 0.5)
                                                        hrs = timeLapsed/3600.
                                                        t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
                                                        print(t_str)                
                                                        # f_res.write("%s\n" % t_str)

# f_res.close()
f_tsv.close()
