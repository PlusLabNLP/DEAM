import sys
import argparse
import numpy as np
import itertools
import random
import os

random.seed(1227)
np.random.seed(1227)

class Manipulation():

	"""
		ind: which conversation
		idx: in each conversation, which utterance
	"""

	def __init__(self, fname):
		'''
		Param:
			fname: input file 
		'''
		self.whole_convs=self.load_dataset(fname)
		self.num_convs=len(self.whole_convs)
		print('all {} convs amrs are loaded'.format(self.num_convs))

	def load_dataset(self, fname):
		'''load input file
		Param:
			fname: input file 
		'''
		fr = open(fname, 'r')
		return fr.readlines()

	def load_conversation(self, ind):
		'''load a specific conversation
		Param:
			ind: conversation's index
		'''
		self.conv_ind = ind
		if '</CONV>' in self.whole_convs[ind]:
			self.conv_utts = self.whole_convs[ind].strip().split('</CONV>')[1].split('</UTT>')
		else:
			self.conv_utts = self.whole_convs[ind].strip().split('</UTT>')
		return self.conv_utts

	def get_conversation(self, ind):
		'''get a conversation's utterances
		Param:
			ind: conversation's index
		'''
		if '</CONV>' in self.whole_convs[ind]:
			return self.whole_convs[ind].strip().split('</CONV>')[1].split('</UTT>')
		else:
			return self.whole_convs[ind].strip().split('</UTT>')

	def get_random_utt_not_from_the_same_conv(self, not_from_ind):
		'''randomly select an utterance not from the same conversation
		Param:
			not_from_ind: conversation's index to be excluded
		'''
		ind_pool = list(range(not_from_ind)) + list(range(not_from_ind+1,self.num_convs))
		ind_picked = np.random.choice(ind_pool)
		utt_pool = self.get_conversation(ind_picked)
		idx_picked = np.random.choice(list(range(len(utt_pool))))
		random_utt = utt_pool[idx_picked]
		return ind_picked, idx_picked, random_utt

	def utterance_ordering(self, replace=False):
		'''change the order of utterances
		Param:
			replace: replace a randomly ordered utterances in the conversation with its original utterances
		'''
		order_permuted_idxes = np.random.permutation(len(self.conv_utts))
		self.neg_uo_utts = [self.conv_utts[idx] for idx in order_permuted_idxes]
		if replace:
			self.conv_utts = self.neg_uo_utts
		return self.neg_uo_utts

	def even_utterance_ordering(self, replace=False):
		'''change the order of utterances of one speaker
		Param:
			replace: replace a randomly ordered utterances in the conversation with its original utterances
		'''
		print(self.conv_utts)
		num_utts = len(self.conv_utts)
		idxes_one_speaker = list(range(0, num_utts, 2))
		idxes_other_speaker = list(range(1, num_utts, 2))
		if np.random.rand() > 0.5:
			speaker_idxes_to_shuffle = idxes_one_speaker
			np.random.shuffle(speaker_idxes_to_shuffle)
			order_permuted_idxes = [j for i in zip(speaker_idxes_to_shuffle,idxes_other_speaker) for j in i]
			if len(speaker_idxes_to_shuffle)>len(idxes_other_speaker):
				order_permuted_idxes += speaker_idxes_to_shuffle[len(idxes_other_speaker):]
			elif len(speaker_idxes_to_shuffle) < len(idxes_other_speaker):
				order_permuted_idxes += idxes_other_speaker[len(speaker_idxes_to_shuffle):]
		else:
			speaker_idxes_to_shuffle = idxes_other_speaker
			np.random.shuffle(speaker_idxes_to_shuffle)
			order_permuted_idxes = [j for i in zip(idxes_one_speaker,speaker_idxes_to_shuffle) for j in i]
			if len(speaker_idxes_to_shuffle) > len(idxes_one_speaker):
				order_permuted_idxes += speaker_idxes_to_shuffle[len(idxess_one_speaker):]
			elif len(speaker_idxes_to_shuffle) < len(idxes_one_speaker):
				order_permuted_idxes += idxes_one_speaker[len(speaker_idxes_to_shuffle):]
		
		self.neg_even_uo_utts = [self.conv_utts[idx] for idx in order_permuted_idxes]
		if replace:
			self.conv_utts = self.neg_even_uo_utts
		return self.neg_even_uo_utts

	def utterance_replacement(self, replace=False):
		'''replace an utterance with a random one
		Param:
			replace: replace a randomly selected utterance with its original one
		'''
		_, _, random_utt = self.get_random_utt_not_from_the_same_conv(self.conv_ind)
		random_pos = np.random.choice(list(range(len(self.conv_utts))))
		self.neg_ur_utts = self.conv_utts[:random_pos] + [random_utt] + self.conv_utts[random_pos+1:]
		if replace:
			self.conv_utts = self.neg_ur_utts
		return self.neg_ur_utts

	def utterance_insertion(self, replace=False):
		'''remove and then insert a random utterance 
		Param:
			replace: insert a randomly selected utterance with the original one
		'''
		# remove
		num_utts = len(self.conv_utts)
		idx_removed = np.random.choice(list(range(num_utts)))
		utt_removed = self.conv_utts[idx_removed]
		remaining_idxes = list(range(num_utts))
		remaining_idxes.remove(idx_removed)
		remaining_utts = [self.conv_utts[idx] for idx in remaining_idxes]
		remaining_utts.append("")
		# re-insert
		idx_to_reinsert = np.random.choice(remaining_idxes)
		remaining_utts[idx_to_reinsert+1:] = remaining_utts[idx_to_reinsert:-1]
		remaining_utts[idx_to_reinsert] = utt_removed
		self.neg_ui_utts = remaining_utts
		if replace:
			self.conv_utts = self.neg_ui_utts
		return self.neg_ui_utts

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='create coherence data')
	parser.add_argument('--which_split', type=str, default='train',
		help='type of data to create its negative samples train/valid/test.')
	parser.add_argument('--data_folder_path', type=str, default='data/topical_persona',
		help='the name of the folder (not path) where train/valid/test original data is.')
	args = parser.parse_args()


	fname = '{}/{}.txt'.format(args.data_folder_path, args.which_split)
	man_mesgar = Manipulation(fname)

	# create folder if not exist
	folder_path = '{}/manipulation_mesgar/'.format(args.data_folder_path)
	isExist = os.path.exists(folder_path)
	if not isExist:
		os.makedirs(folder_path)
	foutput = '{}/manipulation_mesgar/{}.txt'.format(args.data_folder_path, args.which_split)
	fw = open(foutput, 'w')
	foutput_mplts = '{}/manipulation_mesgar/{}_mplts.txt'.format(args.data_folder_path, args.which_split)
	fw_mplts = open(foutput_mplts, 'w')

	# randomly select 1 to all 4 different type of manipulations
	type_manipulations = ['utterance_ordering', 'utterance_insertion', 'utterance_replacement', 'even_utterance_ordering']
	num_chgs = [i for i in range(1, len(type_manipulations)+1)]

	for ind in range(man_mesgar.num_convs):

		man_mesgar.load_conversation(ind)

		fw.write('</UTT>'.join(man_mesgar.conv_utts) + '</UTT>1\n')

		print('\n\nconv {}th'.format(ind))
		print('ORIGINAL UTTERANCES:')
		for ind, sent in enumerate(man_mesgar.conv_utts):
			print(ind, sent)

		num_manipulations = np.random.choice(num_chgs)
		manipulations = np.random.choice(type_manipulations, num_manipulations, replace=False)
	

		if 'utterance_ordering' in manipulations:
			man_mesgar.utterance_ordering(replace=True)
			print('************************utterance_ordering******************************')
			print('UPDATED UTTERANCES AFFTER utterance_ordering:')
			for ind, sent in enumerate(man_mesgar.conv_utts):
				print(ind, sent)

		if 'utterance_insertion' in manipulations:
			man_mesgar.utterance_insertion(replace=True)
			print('************************utterance_insertion******************************')
			print('UPDATED UTTERANCES AFFTER utterance_insertion:')
			for ind, sent in enumerate(man_mesgar.conv_utts):
				print(ind, sent)

		if 'even_utterance_ordering' in manipulations:
			man_mesgar.even_utterance_ordering(replace=True)
			print('************************even_utterance_ordering******************************')
			print('UPDATED UTTERANCES AFFTER even_utterance_ordering:')
			for ind, sent in enumerate(man_mesgar.conv_utts):
				print(ind, sent)

		if 'utterance_replacement' in manipulations:
			man_mesgar.utterance_replacement(replace=True)
			print('************************utterance_replacement******************************')
			print('UPDATED UTTERANCES AFFTER utterance_replacement:')
			for ind, sent in enumerate(man_mesgar.conv_utts):
				print(ind, sent)

		for ind, sent in enumerate(man_mesgar.conv_utts):
			print(ind, sent)
		fw.write('</UTT>'.join(man_mesgar.conv_utts) + '</UTT>0\n')
		fw_mplts.write(str(manipulations)+'\n')
