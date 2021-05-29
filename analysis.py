import csv
from sklearn.metrics import f1_score
import numpy as np

def testdata_analysis(pred_labs,true_labs,ids,filename):
 	with open(filename, 'r') as csvfile:
	    reader = csv.DictReader(csvfile, delimiter = '\t')
	    twitter_count = 0
	    gab_count = 0
	    file= open("ids.txt","w")
	    index = 6978
	    for ind,row in enumerate(reader):
	    	if row['source'] == "twitter":
	    		if pred_labs[ind] == true_labs[ind]:
	    			twitter_count = twitter_count + 1
	    			file.write(str(ind+index))
	    			file.write('\n')
	    	else:
	    		if pred_labs[ind] == true_labs[ind]:
	    			gab_count = gab_count + 1
	    			file.write(str(ind+index))
	    			file.write('\n')
	    print (twitter_count)
	    print (gab_count)

def insights_results_lab(pred_vals, true_vals, num_runs,NUM_CLASSES,model_name,task,conf_map):

	dyn_fname_lab = ("fscore_%s_%s.txt" % (task, model_name))
	f_err_lab = open(dyn_fname_lab, 'w')
	f_err_lab.write("lab id\tlabel\tF score\n")
	class_acc ={}
	for key in range(NUM_CLASSES):
		class_acc[key] = []
	for i in range(num_runs):
		arr = f1_score(true_vals[i], pred_vals[i], average=None)
		for class_ind in range(NUM_CLASSES):
			class_acc[class_ind].append([arr[class_ind]])
	for num in range(len(class_acc)):
		mean_runs = np.mean(class_acc[num], axis = 0)
		f_err_lab.write("%d\t%s\t%.3f\n" % (num, conf_map['FOR_LMAP'][num],mean_runs [0]))
	f_err_lab.close()