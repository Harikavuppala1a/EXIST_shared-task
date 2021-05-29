import os
import numpy as np
import csv
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

def rec_label(tp, fn):
	if (tp + fn) == 0:
		# print "denominator zero"
		return 1
	return tp/float(tp + fn)

def prec_label(tp, fp):
	if (tp + fp) == 0:
		# print "denominator zero"
		return 1
	return tp/float(tp + fp)

def f_label(tp, fp, fn):
	if (2*tp + fn + fp) == 0:
		# print "denominator zero"
		return 1
	return (2*tp)/float(2*tp + fn + fp)

def components_F(pred, act, NUM_CLASSES):
	TP = np.zeros(NUM_CLASSES)
	FP = np.zeros(NUM_CLASSES)
	# TN = np.zeros(NUM_CLASSES)
	FN = np.zeros(NUM_CLASSES)
	for l_id in range(NUM_CLASSES):
		for pr, ac in zip(pred, act):
			if l_id == ac:
				if l_id == pr:
					TP[l_id] += 1
				else:
					FN[l_id] += 1
			else:
				if l_id == pr:
					FP[l_id] += 1
				# else:
				# 	TN[l_id] += 1
	return TP, FP, FN	

def insights_results_lab_one_run(pred_vals, true_vals, train_labels, dyn_fname_part, out_fold, num_runs,NUM_CLASSES, FOR_LMAP):
	TP, FP, FN = components_F(pred_vals, true_vals, NUM_CLASSES)
	train_coverage = np.zeros(NUM_CLASSES)
	for lset in train_labels:
		# for l in lset:
			train_coverage[lset] += 1.0
	train_coverage /= float(len(train_labels))

	dyn_fname_lab = ("%slab/%s.txt" % (out_fold, dyn_fname_part))
	f_err_lab = open(dyn_fname_lab, 'w')
	f_err_lab.write("lab id\tlabel\ttrain cov\tPrec\tRecall\tF score\n")
	class_ind = 0
	arr = f1_score(true_vals,pred_vals,average=None)
	for tp, fp, fn in zip(TP,FP,FN):
		P = prec_label(tp, fp)
		R = rec_label(tp, fn)
		F = f_label(tp, fp, fn)
		f_err_lab.write("%d\t%s\t%.2f\t%.3f\t%.3f\t%.3f\n" % (class_ind, FOR_LMAP[class_ind],train_coverage[class_ind]*100,0,0,arr[class_ind]))
		class_ind += 1
	f_err_lab.close()

def insights_results_lab(pred_vals_across_runs, true_vals, train_labels, dyn_fname_part, out_fold, num_runs, NUM_CLASSES, FOR_LMAP):
	inst_fold_str = ("%slab/" % (out_fold))
	os.makedirs(inst_fold_str, exist_ok=True)
	dyn_fname_lab = ("%s%s.txt" % (inst_fold_str, dyn_fname_part))
	f_err_lab = open(dyn_fname_lab, 'w')
	f_err_lab.write("lab id\tlabel\ttrain cov\tF score\n")
	train_coverage = np.zeros(NUM_CLASSES)
	for lset in train_labels:
		# for l in lset:
			train_coverage[lset] += 1.0
	train_coverage /= float(len(train_labels))
	class_acc = {}
	for key in range(NUM_CLASSES):
		class_acc[key] = []
	for i in range(num_runs):
		arr = f1_score(true_vals,pred_vals_across_runs[i],average=None)
		for class_ind in range(NUM_CLASSES):
			class_acc[class_ind].append([arr[class_ind]])
	for num in range(len(class_acc)):
		mean_runs = np.mean(class_acc[num],axis =0)
		f_err_lab.write("%d\t%s\t%.2f\t%.3f\n" % (num, FOR_LMAP[num],train_coverage[num]*100,mean_runs[0]))
	f_err_lab.close()

def testdata_analysis(pred_vals_across_runs, true_vals):
	with open("EXIST2021_test_labeled.tsv", 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter = '\t')
		twitter_count = 0
		gab_count = 0
		file= open("ids_sdbaseline.txt","w")
		index = 6978
		for ind,row in enumerate(reader):
			if row['source'] == "twitter":
				# print (pred_vals_across_runs[ind])
				# print (true_vals[ind])
				# break
				if pred_vals_across_runs[ind][0] == true_vals[ind]:
					twitter_count = twitter_count + 1
					file.write(str(ind+index))
					file.write('\n')
			else:
				if pred_vals_across_runs[ind][0] == true_vals[ind]:
					gab_count = gab_count + 1
					file.write(str(ind+index))
					file.write('\n')
		print (twitter_count)
		print (gab_count)

def write_results(lang,word_feat_str,sent_enc_feat_str,classimb,use_emotions, use_hashtags, use_empath, use_perspective, use_hurtlex, metr_dict, f_tsv, prob_type, conf_dict_com,rnn_dim,att_dim):
	if prob_type == 'multi-class':
		f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%.3f\t%.3f\t%.3f\n" % (lang, word_feat_str,sent_enc_feat_str,classimb,use_emotions, use_hashtags, use_empath,use_perspective, use_hurtlex, rnn_dim,att_dim, metr_dict['avg_f_we'],metr_dict['avg_f_ma'],metr_dict['avg_f_mi'],metr_dict['avg_acc'],metr_dict['avg_p_we'],metr_dict['avg_p_ma'],metr_dict['avg_p_mi'],metr_dict['avg_r_we'],metr_dict['avg_r_ma'],metr_dict['avg_r_mi'],metr_dict['std_f_we'],conf_dict_com["test_mode"], conf_dict_com['LEARN_RATE'], conf_dict_com['dropO1'], conf_dict_com['dropO2']))
	elif prob_type == 'binary':
		f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%.3f\t%.3f\t%.3f\n" % (lang, word_feat_str,sent_enc_feat_str,classimb,use_emotions, use_hashtags,use_empath, use_perspective, use_hurtlex, rnn_dim,att_dim, metr_dict['avg_f'],metr_dict['avg_p'],metr_dict['avg_r'],metr_dict['avg_acc'],metr_dict['std_f'],metr_dict['std_acc'],conf_dict_com["test_mode"], conf_dict_com['LEARN_RATE'], conf_dict_com['dropO1'], conf_dict_com['dropO2']))
	lines = prep_write_lines(metr_dict, prob_type)
	for line in lines:
		print(line)
		# f_res.write(line + '\n')
