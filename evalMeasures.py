import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

metrics = {
	'multi-class': ['acc', 'p_mi', 'r_mi', 'f_mi', 'p_ma', 'r_ma', 'f_ma', 'p_we', 'r_we', 'f_we'],
	'binary': ['acc', 'p', 'r', 'f'],
}

def init_metr_dict(prob_type):
	global metrics
	metr_dict = {}
	for key in metrics[prob_type]:
		metr_dict[key] = []
	return metr_dict


def aggregate_metr(metr_dict, num_vals, prob_type):
	global metrics
	for key in metrics[prob_type]:
		s = 0
		s_sq = 0
		for v in metr_dict[key]:
			s += v
			s_sq += v**2
		avg_v = s/num_vals
		metr_dict['avg_' + key] = avg_v
		metr_dict['std_' + key] = np.sqrt(s_sq/num_vals - avg_v**2)
		metr_dict[key] = []
	return metr_dict

def calc_metrics_print(pred_vals_sc, true_vals_sc, metr_dict,NUM_CLASSES,prob_type):
	if prob_type == 'multi-class':
		# pred_vals_sc = [labels for labels in pred_vals]
		# true_vals_sc = [labels for labels in true_vals]
		metr_dict['p_mi'].append(precision_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='micro'))
		metr_dict['r_mi'].append(recall_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='micro'))
		metr_dict['f_mi'].append(f1_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='micro'))
		metr_dict['p_ma'].append(precision_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='macro'))
		metr_dict['r_ma'].append(recall_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='macro'))
		metr_dict['f_ma'].append(f1_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='macro'))
		metr_dict['p_we'].append(precision_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='weighted'))
		metr_dict['r_we'].append(recall_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='weighted'))
		metr_dict['f_we'].append(f1_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='weighted'))
		metr_dict['acc'].append(accuracy_score(true_vals_sc, pred_vals_sc))
	elif prob_type == 'binary':
		# pred_vals_sc = [labels for labels in pred_vals]
		# true_vals_sc = [labels for labels in true_vals]
		metr_dict['p'].append(precision_score(true_vals_sc, pred_vals_sc))
		metr_dict['r'].append(recall_score(true_vals_sc, pred_vals_sc))
		metr_dict['f'].append(f1_score(true_vals_sc, pred_vals_sc))
		metr_dict['acc'].append(accuracy_score(true_vals_sc, pred_vals_sc))
	
	return metr_dict

def prep_write_lines(metr_dict, prob_type):
	if prob_type == 'multi-class':
		lines = [
			"f1 Weighted: %.3f, std: %.3f" % (metr_dict['avg_f_we'], metr_dict['std_f_we']),
			"f1 Macro: %.3f" % metr_dict['avg_f_ma'],
			"f1 Micro: %.3f" % metr_dict['avg_f_mi'],
			"P Weighted: %.3f" % metr_dict['avg_p_we'],
			"R Weighted: %.3f" % metr_dict['avg_r_we'],
			"Accuracy: %.3f" % metr_dict['avg_acc'],
		]
	elif prob_type == 'binary':
		lines = [
			"f1: %.3f, std: %.3f" % (metr_dict['avg_f'], metr_dict['std_f']),
			"P: %.3f" % metr_dict['avg_p'],
			"R: %.3f" % metr_dict['avg_r'],
			"Accuracy: %.3f, std: %.3f" % (metr_dict['avg_acc'], metr_dict['std_acc']),
		]
	return lines

def write_results(lang,original_text,classimb,feat_type,model_name,use_emotions, use_hashtags, use_empath, use_perspective, use_hurtlex, metr_dict, f_tsv, prob_type, conf_dict_com,rnn_dim,att_dim):
	if prob_type == 'multi-class':
		f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%.3f\t%.3f\t%.3f\n" % (lang,original_text, classimb, feat_type ,model_name,use_emotions, use_hashtags, use_empath,use_perspective, use_hurtlex, rnn_dim,att_dim, metr_dict['avg_f_we'],metr_dict['avg_f_ma'],metr_dict['avg_f_mi'],metr_dict['avg_acc'],metr_dict['avg_p_we'],metr_dict['avg_p_ma'],metr_dict['avg_p_mi'],metr_dict['avg_r_we'],metr_dict['avg_r_ma'],metr_dict['avg_r_mi'],metr_dict['std_f_we'],conf_dict_com["test_mode"], conf_dict_com['LEARN_RATE'], conf_dict_com['dropO1'], conf_dict_com['dropO2']))
	elif prob_type == 'binary':
		f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%.3f\t%.3f\t%.3f\n" % (lang, original_text, classimb, feat_type,model_name,use_emotions, use_hashtags,use_empath, use_perspective, use_hurtlex, rnn_dim,att_dim, metr_dict['avg_f'],metr_dict['avg_p'],metr_dict['avg_r'],metr_dict['avg_acc'],metr_dict['std_f'],metr_dict['std_acc'],conf_dict_com["test_mode"], conf_dict_com['LEARN_RATE'], conf_dict_com['dropO1'], conf_dict_com['dropO2']))
	lines = prep_write_lines(metr_dict, prob_type)
	for line in lines:
		print(line)
		# f_res.write(line + '\n')