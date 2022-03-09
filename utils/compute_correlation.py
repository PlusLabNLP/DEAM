import argparse
import csv
import os
from scipy.stats import spearmanr
import json

if __name__=='__main__':
	#Retrieve ground-truth and predicted scores and compute their correlations
	parser = argparse.ArgumentParser(description='Create tsv files for coherence train/valid/test sets')
	parser.add_argument('--model_path', type=str, required=True, help='directory of the model\'s predicted scores')

	args = parser.parse_args()

	tests = {
		'fed_test_coherence_orig': ("./data/fed", 'fed_test_coherence.txt'),
		'fed_test_overall_orig': ("./data/fed", 'fed_test_overall.txt'),
		'dstc9_test_coherence_orig': ("./data/dstc9", 'dstc9_test_coherence_averaged.txt'),
		'dstc9_test_overall_orig': ("./data/dstc9", 'dstc9_test_overall_averaged.txt'),
		}

	results = {test_name: -1 for test_name in tests}

	for test_name in tests:
		ground_truth_file = os.path.join(*tests[test_name])
		fr1 = open(ground_truth_file, 'r')
		gt_lines=fr1.readlines()
		gt_scores=[]
		
		prediction_file = os.path.join(args.model_path, test_name+"_preds.txt")
		fr2= open(prediction_file, 'r')
		pred_lines=fr2.readlines()
		pred_scores=[]
		
		# if "dstc9" not in test_name: 
		for ind, gt_line in enumerate(gt_lines):
			gt_scores.append(float(gt_line.split('\n')[0].split('</UTT>')[-1]))
			pred_scores.append(float(pred_lines[ind].split('\n')[0].split('</UTT>')[-1]))
		
		sp_score=spearmanr(gt_scores, pred_scores)
		results[test_name] = sp_score
		print("###Spearman Correlation {}:{}###".format(test_name,sp_score[0]))
	json.dump(results, open(os.path.join(args.model_path, "coherence_results.json"), 'w'), indent=4)
