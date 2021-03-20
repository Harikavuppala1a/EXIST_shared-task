import os
import numpy as np
from dl_models import get_model, attLayer_hier, br_binary_loss, lp_categ_loss
from sklearn.utils import class_weight
from loadPreProc import *
from evalMeasures import *
from keras.utils import to_categorical
from keras import backend as K
from gen_batch_keras import TrainGenerator, TestGenerator
import pickle
import json

def evaluate_model(mod_op_list, data_dict, bac_map, prob_trans_type, metr_dict, thresh, att_flag, output_folder_name, fname_part_r_ind, classi_probs_label_info):
    y_pred_list = []
    true_vals = data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']]
    num_test_samp = len(true_vals)
    sum_br_lists = np.zeros(num_test_samp, dtype=np.int64)
    arg_max_br_lists = np.empty(num_test_samp, dtype=np.int64)
    max_br_lists = np.zeros(num_test_samp)
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val
        if prob_trans_type == 'multi-class':
            y_pred  = np.argmax(mod_op, 1)
        elif prob_trans_type == "binary":
            mod_op = np.squeeze(mod_op, -1)
            y_pred = np.rint(mod_op).astype(int)
        y_pred_list.append(y_pred)

    if prob_trans_type == 'multi-class':
        pred_vals = bin_classi_op_to_label_lists(y_pred_list[0])
    elif prob_trans_type == "binary":
            pred_vals = bin_classi_op_to_label_lists(y_pred_list[0])

    return pred_vals, true_vals, calc_metrics_print(pred_vals, true_vals, metr_dict, data_dict['NUM_CLASSES'], data_dict['prob_type'])

def train_predict(word_feats, sent_enc_feats, trainY, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, class_w, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set, m_ind, run_ind, save_folder_name, use_saved_model, gen_att, learn_rate, dropO1, dropO2, batch_size, num_epochs, save_model, save_trained_mod,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable, use_emotions, emoji_array, use_hashtags, hashtag_array):
    att_op = None
    fname_mod_op = ("%s%s/iop~%d~%d.pickle" % (save_folder_name, fname_part, m_ind, run_ind))
    if use_saved_model and os.path.isfile(fname_mod_op):
        print("loading model o/p")
        with open(fname_mod_op, 'rb') as f:
            mod_op = pickle.load(f)
        if gen_att:
            fname_att_op = ("%s%s/att_op~%d~%d.pickle" % (save_folder_name, fname_part, m_ind, run_ind))
            if os.path.isfile(fname_att_op):
                with open(fname_att_op, 'rb') as f:
                    att_op = pickle.load(f)
    else:
        model, att_mod = get_model(model_type, data_dict['max_post_length'], data_dict['max_num_sent'], data_dict['max_words_sent'], word_feats, sent_enc_feats, learn_rate, dropO1, dropO2, num_cnn_filters, rnn_type, loss_func, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable, use_emotions, emoji_array, use_hashtags, hashtag_array)
        training_generator = TrainGenerator(np.arange(0, data_dict['train_en_ind']), trainY, word_feats, sent_enc_feats, data_dict, batch_size,use_emotions,emoji_array,use_hashtags, hashtag_array)
        model.fit_generator(generator=training_generator, epochs=num_epochs, shuffle=False, verbose=1, use_multiprocessing=False, workers=1)
        test_generator = TestGenerator(np.arange(data_dict['test_st_ind'], data_dict['test_en_ind']), word_feats, sent_enc_feats, data_dict, batch_size,use_emotions,emoji_array, use_hashtags, hashtag_array)
        mod_op = model.predict_generator(generator=test_generator, verbose=1, use_multiprocessing=False, workers=1)
        if gen_att and (att_mod is not None):
            att_op = att_mod.predict_generator(generator=test_generator, verbose=1, use_multiprocessing=False, workers=1)
            if type(att_op) != list:
                att_op = [att_op]

        if save_model:    
            print("saving model o/p")
            os.makedirs(save_folder_name + fname_part, exist_ok=True)
            with open(fname_mod_op, 'wb') as f:
                pickle.dump(mod_op, f)
            if run_ind == 0:
                if save_trained_mod:
                    fname_mod = ("%s%s/mod~%d.h5" % (save_folder_name, fname_part, m_ind))
                    model.save(fname_mod)
                if m_ind == 0:
                    with open("%s%s/mod_sum.txt" % (save_folder_name, fname_part),'w') as fh:
                        model.summary(print_fn=lambda x: fh.write(x + '\n'))                                                                
            if gen_att:
                fname_att_op = ("%s%s/att_op~%d~%d.pickle" % (save_folder_name, fname_part, m_ind, run_ind))
                with open(fname_att_op, 'wb') as f:
                    pickle.dump(att_op, f)
        K.clear_session()
    return mod_op, att_op

def class_imb_loss_nonlin(trainY_noncat_list, class_imb_flag, num_classes_list, prob_trans_type, test_mode, save_fold_path, use_saved_data_stuff, save_data_stuff):
    filename = "%sclass_imb~%s~%s~%s.pickle" % (save_fold_path, class_imb_flag, prob_trans_type, test_mode)
    if use_saved_data_stuff and os.path.isfile(filename):
        print("loading class imb for %s and %s; test mode = %s" % (class_imb_flag, prob_trans_type, test_mode))
        with open(filename, 'rb') as f:
            nonlin, out_vec_size_list, cw_list = pickle.load(f)
    else:
        if prob_trans_type == "multi-class": 
            nonlin = 'softmax'
            out_vec_size_list = num_classes_list
            cw_list = [None]
        elif prob_trans_type == "binary":    
            nonlin = 'sigmoid'
            out_vec_size_list = [1]*len(trainY_noncat_list)
            cw_list = [None]*len(trainY_noncat_list)

        if class_imb_flag:
            cw_list = []
            if prob_trans_type == "multi-class" or prob_trans_type == "binary": 
                for num_classes_var, trainY_noncat in zip(num_classes_list, trainY_noncat_list):
                    tr_uniq = np.arange(num_classes_var)
                    cw_arr = class_weight.compute_class_weight('balanced', tr_uniq, trainY_noncat)
                    cw_list.append(cw_arr)

        if save_data_stuff:        
            print("saving class imb for %s and %s; test mode = %s" % (class_imb_flag, prob_trans_type, test_mode))
            with open(filename, 'wb') as f:
                pickle.dump([nonlin, out_vec_size_list, cw_list], f)

    if class_imb_flag:
        loss_func_list = []
        for cw_arr in cw_list:
            if prob_trans_type == "multi-class":
                loss_func_list.append(lp_categ_loss(cw_arr))
            elif prob_trans_type == "binary":    
                loss_func_list.append(br_binary_loss(cw_arr))
    else:
        if prob_trans_type == "multi-class":
            loss_func_list = ['categorical_crossentropy']
        elif prob_trans_type == "binary":    
            loss_func_list = ['binary_crossentropy']*len(trainY_noncat_list)

    return loss_func_list, nonlin, out_vec_size_list, cw_list

def transform_labels(data_trainY, prob_trans_type, test_mode, save_fold_path, NUM_CLASSES, data_prob_type, classi_probs_label_info, classi_probs_label_str, use_saved_data_stuff, save_data_stuff):
    filename = "%slabel_info~%s~%s~%s.pickle" % (save_fold_path, prob_trans_type, classi_probs_label_str, test_mode)
    if use_saved_data_stuff and os.path.isfile(filename):
        print("loading label info for %s; classi_probs_label_info = %s, test mode = %s" % (prob_trans_type, classi_probs_label_str, test_mode))
        with open(filename, 'rb') as f:
            trainY_list, trainY_noncat_list, num_classes_list = pickle.load(f)
    else:
        if prob_trans_type == "multi-class":
            num_classes_var = 6
            bac_map = None
            data_train_new = []
            for i in data_trainY:
                data_train_new.append([i])
            trainY = trans_labels_multi_classi(data_train_new)
            trainY_noncat_list = [trainY]
            trainY_list = [to_categorical(trainY, num_classes=num_classes_var)]
            num_classes_list = [6]
        elif prob_trans_type == "binary":           
            trainY_list = trans_labels_bin_classi(data_trainY)
            num_classes_list = [2]
            bac_map = None
            trainY_noncat_list = list(trainY_list)
        if save_data_stuff:        
            print("saving label info for %s; classi_probs_label_info = %s, test mode = %s" % (prob_trans_type, classi_probs_label_str, test_mode))
            with open(filename, 'wb') as f:
                pickle.dump([trainY_list, trainY_noncat_list, num_classes_list], f)
    return trainY_list, trainY_noncat_list, num_classes_list,None
