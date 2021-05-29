import numpy as np
from sklearn.model_selection import train_test_split
from loadPreProc import *
from evalMeasures import *
import sys

conf_dict_list, conf_dict_com = load_config(sys.argv[1])

if conf_dict_com['prob_type'] == "multi-class":
   tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_mc_filename"]
else:
    tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_b_filename"]
data_dict = load_data(conf_dict_com['filename'], conf_dict_com['data_folder_name'], conf_dict_com['data_train_name'], conf_dict_com['data_test_name'], conf_dict_com['save_folder_name'], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['language'],  conf_dict_com['prob_type'], conf_dict_com['filename_map_list'],conf_dict_com['test_mode'])
print (len(data_dict))

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
# print (data_dict['lab'][5])
if os.path.isfile(tsv_path):
    f_tsv = open(tsv_path, 'a')
else:
    f_tsv = open(tsv_path, 'w')
    if conf_dict_com['prob_type'] == "multi-class":
        f_tsv.write("language\tclass_imb\tfeature\tuse_emotions\tuse_hashtags\tuse_empath\tuse_perpective\tuse_hurtlex\trnn_dim\tatt_dim\tavg_f_we\tavg_f_ma\tavg_f_mi\tavg_acc\tavg_p_we\tavg_p_ma\tavg_p_mi\tavg_r_we\tavg_r_ma\tavg_r_mi\tstd_f_we\ttest_mode\tlearnrate\tdrop1\tdrop2\n")
    else:
        f_tsv.write("language\tclass_imb\tfeature\tuse_emotions\tuse_hashtags\tuse_empath\tuse_perpective\tuse_hurtlex\trnn_dim\tatt_dim\tavg_f\tavg_p\tavg_r\tavg_ac\tstd_f\ttest_mode\tlearnrate\tdrop1\tdrop2\n")

def train_evaluate_model(trainY_list, true_vals, metr_dict, prob_type, NUM_CLASSES):
	train_coverage = np.zeros(NUM_CLASSES)
	for lset in trainY_list:
		# print (lset)
		# for l in lset:
			train_coverage[lset] += 1.0
	train_coverage /= float(len(trainY_list))
	print(train_coverage)
	print(np.mean(train_coverage))

	pred_vals = []
	for i in range(len(true_vals)):
		r_num = np.random.uniform()
		start_range = 0
		for j in range(NUM_CLASSES):
			end_range = start_range + train_coverage[j]
			if r_num >= start_range and r_num < end_range:
				pred_vals.append([j])
				break
			start_range = end_range

	return pred_vals, true_vals, calc_metrics_print(pred_vals, true_vals, metr_dict, NUM_CLASSES,prob_type)


metr_dict = init_metr_dict(conf_dict_com['prob_type'])
for run_ind in range(conf_dict_com["num_runs"]):
	pred_vals, true_vals, metr_dict = train_evaluate_model(data_dict['lab'][:data_dict['train_en_ind']], data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']], metr_dict, data_dict['prob_type'], data_dict['NUM_CLASSES'])

metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs"], data_dict['prob_type'])
write_results(conf_dict_com['language'], '','','',conf_dict_com['use_emotions'],conf_dict_com['use_hashtags'],conf_dict_com['use_empathfeats'], conf_dict_com['use_perspectivefeats'], conf_dict_com['use_hurtlexfeats'], metr_dict,f_tsv, conf_dict_com['prob_type'],conf_dict_com,'','')




