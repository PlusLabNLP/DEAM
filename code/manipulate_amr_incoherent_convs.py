import amrlib
import numpy as np
import re
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
import argparse
import os
import random
random.seed(1234)
np.random.seed(1000)

#loading amr-text and text-amr models
stog = amrlib.load_stog_model(model_dir='amr_models/model_parse_t5-v0_1_0', device='cuda:0')
gtos = amrlib.load_gtos_model('amr_models/model_generate_t5-v0_1_0', device='cuda:0')



class Manipulation():
	def __init__(self, fname, args, which_split):
		
		self.args=args
		self.which_split = which_split
		self.whole_convs=self.load_dataset(fname)
		self.num_convs=len(self.whole_convs)
		print('{} conversations with their amrs are loaded'.format(self.num_convs))

		self.antonyms={}
		self.load_antonyms("utils/conceptnet_antonyms.txt", self.antonyms)

		self.synonyms = {}
		self.load_synonyms("utils/conceptnet_synonyms.txt", self.synonyms)



	def load_dataset(self, fname):
		'''load dataset including input conversations
		Param:
			fname: input file 
		'''
		fr = open(fname, 'r')
		return fr.readlines()



	def load_specific_utt_amrs(self, idx):
		'''load amr of a specific conversation
		Param:
			ind: the index of a conversation to load its amr
		'''
		amrs=self.whole_convs[idx].split('</UTT>')
		self.utts_amrs=[]
		self.curr_idx = idx
		self.utts=[]
		self.conv_idx_utt_ind_tuples_with_most_overlapping_entities = None
		for ind, amr in enumerate(amrs):
			parts = amr.split('\\n')
			self.utts_amrs.append('\n'.join(parts[1:]))
			self.utts.append(parts[0].split('# ::snt')[1].strip())



	def load_antonyms(self, txt_path, antonyms):
		'''load antonyms of concepts from ConceptNet
		Param:
			txt_path: the path including antonyms relations from ConceptNet 
			antonyms: a dictionary of concepts and their antonyms 
		'''
		with open(txt_path, 'r') as fr:
			for cpt_rel_ant in fr.readlines():
				cpt, rel, ant = cpt_rel_ant.split('|||')
				ant = ant.split('\n')[0]
				if rel == '/r/Antonym' and cpt not in antonyms:
					antonyms[cpt]=[ant]
				elif rel == '/r/Antonym' and cpt in antonyms:
					antonyms[cpt].append(ant)



	def load_synonyms(self, txt_path, synonyms):
		'''load synonym of concepts from ConceptNet
		Param:
			txt_path: the path including synonyms relations from ConceptNet 
			synonyms: a dictionary of concepts and their synonyms 
		'''
		with open(txt_path, 'r') as fr:
			for cpt_rel_ant in fr.readlines():
				cpt, rel, ant = cpt_rel_ant.split('|||')
				ant = ant.split('\n')[0]
				if cpt not in synonyms:
					synonyms[cpt] = [ant]
				else:
					synonyms[cpt].append(ant)



	def get_antonym(self, concept):
		'''get one of the randomly selected antonyms of a specified concept
		Param:
			concept: the concept to randomly return one of its  antonyms 
		'''
		return np.random.choice(self.antonyms[concept]) if concept in self.antonyms else None



	def get_synonym(self, concept):
		'''get one of the randomly selected synonyms of a specified concept
		Param:
			concept: the concept to randomly return one of its  synonyms 
		'''
		return np.random.choice(self.synonyms[concept]) if concept in self.synonyms else None


	def utt_repetition(self, selected_ind, insertion_inds):
		'''repeat utterances in specific locations
		Param:
			selected_ind: index of selected to utterances to repeat
			insertion_inds: the locations to insert
		'''
		for ins_ind in insertion_inds:
			if not self.is_multisent(self.utts_amrs[ins_ind]) and not self.is_multisent(self.utts_amrs[selected_ind]):
				#both selected utterance and the one that it wants to append have one sentence and needs to be changed to a multisentence
				new_amr='(m / multi-sentence\n      :snt1 '+self.add_indentation(self.utts_amrs[ins_ind], 1)[:-1]+'\n      :snt2 '+self.add_indentation(self.utts_amrs[selected_ind], 1)
			elif not self.is_multisent(self.utts_amrs[selected_ind]):
				num_sents=self.get_num_sents(self.utts_amrs[ins_ind])
				new_amr=self.utts_amrs[ins_ind][:-1]+'\n      :snt'+str(num_sents+1)+' '+self.add_indentation(self.utts_amrs[selected_ind], 1)
			elif not self.is_multisent(self.utts_amrs[ins_ind]):
				new_amr='(m / multi-sentence\n      :snt1 '+self.add_indentation(self.utts_amrs[ins_ind], 1)[:-1]+self.increase_sents(''.join(self.utts_amrs[selected_ind].split('(m / multi-sentence')[1:]), 2)
			else:
				#both selected utterance and the one that it wants to append are multisentences
				num_sents=self.get_num_sents(self.utts_amrs[ins_ind])
				new_amr=self.utts_amrs[ins_ind][:-1]+self.increase_sents(''.join(self.utts_amrs[selected_ind].split('(m / multi-sentence')[1:]), num_sents+1)
			self.utts_amrs[ins_ind]=new_amr



	def coreference(self, overall_perc=0.5):
		'''Co-Referrence Inconsistency Manipulation
		Param:
			overall_perc: the perentage of manipulations
		'''
		pronouns=['i', 'you', 'he', 'she', 'it', 'we', 'they']

		#get pronouns and their positions in the conv
		conv_pronouns = self.get_pronouns(self.utts_amrs)
		if not conv_pronouns:
			print('This conversation does not include any pronouns to be manipulatedd')
			return

		#select overall_perc% of the pronouns or at least one pronoun in the conversation to be replaced randomly
		num_ind_choose= int(len(conv_pronouns)*overall_perc) if int(len(conv_pronouns)*overall_perc)>0 else 1
		selected_pronouns=np.random.choice(list(conv_pronouns.keys()), size=num_ind_choose, replace=False)

		for selected_pronoun in selected_pronouns:
			opt=np.random.choice((0,1))
			if opt:
				#replace a pronoun with another random pronoun from the same conversation
				replaced_pronoun=np.random.choice(list(set(pronouns)-set(selected_pronouns)), size=1)[0]
				utt_rand_ind=np.random.choice(conv_pronouns[selected_pronoun],1)[0]
				self.utts_amrs[utt_rand_ind]=re.sub('\/ '+selected_pronoun+'\n','/ '+replaced_pronoun+'\n', self.utts_amrs[utt_rand_ind])
				self.utts_amrs[utt_rand_ind]=re.sub('\/ '+selected_pronoun+'\)','/ '+replaced_pronoun+')', self.utts_amrs[utt_rand_ind])
			else:
				#replace a pronoun with another noun from the same conversation
				nouns=self.get_nouns()
				utt_rand_ind=np.random.choice(conv_pronouns[selected_pronoun],1)[0]
				if len(nouns)==0:
					nouns=list(set(pronouns)-set(selected_pronouns))
				replaced_noun=np.random.choice(nouns,1)[0]
				self.utts_amrs[utt_rand_ind]=re.sub('\/ '+selected_pronoun+'\n', '/ '+replaced_noun+'\n', self.utts_amrs[utt_rand_ind])
				self.utts_amrs[utt_rand_ind]=re.sub('\/ '+selected_pronoun+'\)', '/ '+replaced_noun+')', self.utts_amrs[utt_rand_ind])



	def contradiction(self, overall_perc=0.1):
		'''Contradiction Manipulation
		Param:
			overall_perc: the perentage of manipulations
		'''
		#we randomly select overall_perc% of utterances with concepts (not the last two utterances in the conv)
		#then we randomly select to contradict them directly or indirectly and concat them to one randomly selected utterance from the same user
		utt_with_concepts, utt_with_concepts_inds=self.get_utts_with_concepts()
		utt_with_concepts = [u for u,u_id in zip(utt_with_concepts, utt_with_concepts_inds) if u_id<len(self.utts_amrs)-2]
		num_ind_choose= int(len(utt_with_concepts)*overall_perc) if int(len(utt_with_concepts)*overall_perc)>0 else 1
		if not utt_with_concepts:
			print('Warning: there is not concepts in this conv to be manipulated!!')
			return
		selected_utts = np.random.choice(utt_with_concepts, size=num_ind_choose, replace=False)

		for ind in range(len(self.utts_amrs)):
			if self.utts_amrs[ind] in selected_utts:
				#select randomly if we want to do direct contraction or indirect one
				opt=np.random.choice((0,1))
				curr_amr=self.utts_amrs[ind]
				curr_amr_num_sents=self.get_num_sents(curr_amr)
				new_amr=self.direct_negate(ind) if opt==0 else self.indirect_negate(ind)
				#concat the contradiction of the selected utterance (newly resulted amr) to some randomly selected next utterances in the conversation
				user1_2=ind%2
				if user1_2==0:
					#it is an utterance from user1
					utts_insert=[k for k in range(ind+1, len(self.utts_amrs)) if k%2==0]
				else:
					#it is an utterance from user2
					utts_insert=[k for k in range(ind+1, len(self.utts_amrs)) if k%2==1]
				insert_ind = np.random.choice(utts_insert, size=1, replace=False)
				orig_utt=self.utts_amrs[ind]
				self.utts_amrs[ind]=new_amr
				self.utt_repetition(ind, insert_ind)
				self.utts_amrs[ind]=orig_utt

	def partial_irrelevant(self, overall_perc=0.1, replace_chance=0.7):
		'''Irrelevancy Manipulation
		Param:
			overall_perc: the perentage of manipulations
			replace_chance: the perentage of replacement
		'''
		#we randomly select overall_perc% of utterances to make them partially irrelavant to the neighbors
		num_ind_choose= int(len(self.utts_amrs)*overall_perc) if int(len(self.utts_amrs)*overall_perc)>0 else 1
		selected_inds = np.random.choice(len(self.utts_amrs), size=num_ind_choose, replace=False)
		all_items_in_conv=self.get_all_items()
	
		for ind in range(len(self.utts_amrs)):
			if ind in selected_inds:
				utt_items=self.get_utt_items(self.utts_amrs[ind])
				
				#get the number of whole items from different types including concepts, args, ops and mods
				num_items = sum([list(list(utt_items.values())[i]) for i in range(len(utt_items))], [])
				if not num_items:
					print('Warning: this utterance does not have any "concepts, args, ops and mods" to be replaced with some random items of the same conv' )
					continue
				num_items_replace=int(len(num_items)*replace_chance) if int(len(num_items)*replace_chance)>0 else 1
				#randomly select from different types of items
				list_items_choose=[i for i in  ['concept', 'op', 'arg', 'mod'] if utt_items[i] and len(set(all_items_in_conv[i]))>=2]
				while len(list_items_choose)==0:
					ind=np.random.choice(len(self.utts_amrs))
					utt_items=self.get_utt_items(self.utts_amrs[ind])
					list_items_choose=[i for i in  ['concept', 'op', 'arg', 'mod'] if utt_items[i] and len(set(all_items_in_conv[i]))>=2]	
					

				selected_types = np.random.choice(list_items_choose, size=num_items_replace, replace=True)
				for type_item in selected_types:
					selected_item=np.random.choice(list(utt_items[type_item]))
					while True:
						target_item=np.random.choice(list(all_items_in_conv[type_item]))
						if target_item!=selected_item:
							break
					if ' / '+selected_item+'-' in self.utts_amrs[ind]:
						self.utts_amrs[ind]=self.utts_amrs[ind].replace(' / '+selected_item+'-', ' / '+target_item+'-')
					elif ' / '+selected_item+'\n' in self.utts_amrs[ind]:
						self.utts_amrs[ind]=self.utts_amrs[ind].replace(' / '+selected_item+'\n', ' / '+target_item+'\n')
					elif ' / '+selected_item+')' in self.utts_amrs[ind]:
						self.utts_amrs[ind]=self.utts_amrs[ind].replace(' / '+selected_item+')', ' / '+target_item+')')


	def decrease_engagement(self, overall_perc = 0.2):
		'''Decreasing Engagement Manipulation
		Param:
			overall_perc: the perentage of manipulations
		this manipulation is used to decrease the engagingness of the system utterances to represent incoherent convs due to system's not engagingness
		Three types of removal (from high to low priority):
		1. "question_removal" if multi-sentence with question -> remove the question, only remove the question as the last sentence!
		2. "sentence_removal" if multi-sentence with no question -> remove the longest sentence and remove all the sentences after it.
		3. "op_plus_removal" detect the complex substructure(s) in single sentence amrs and randomly choose to remove one and all its sublines
		'''
		num_ind_choose = int(len(self.utts_amrs)*overall_perc)
		#the number of utts that their engagingness will be decreased
		overall_ids_to_select_from = np.random.choice(len(self.utts_amrs), size=num_ind_choose if num_ind_choose > 1 else 1, replace=False)
		multi_sent_amr_ids = [amr_id for amr_id in overall_ids_to_select_from if self.is_multisent(self.utts_amrs[amr_id])]
		amrs_with_ques_amr_ids = [amr_id for amr_id in overall_ids_to_select_from if self.includes_question(self.utts_amrs[amr_id]) and self.is_multisent(self.utts_amrs[amr_id])]
		single_sent_amr_ids = [amr_id for amr_id in overall_ids_to_select_from if amr_id not in multi_sent_amr_ids]
		single_sent_arm_ids_with_2_ops = [amr_id for amr_id in overall_ids_to_select_from if amr_id in single_sent_amr_ids and self.includes_at_least_2_ops(self.utts_amrs[amr_id])]
		single_sent_arm_ids_with_arg = [amr_id for amr_id in overall_ids_to_select_from if amr_id in single_sent_amr_ids and self.includes_arg(self.utts_amrs[amr_id])]
		eligible_utts_ids = multi_sent_amr_ids+amrs_with_ques_amr_ids+single_sent_arm_ids_with_arg
		choosable_ids=[]

		ind=0
		for ind in overall_ids_to_select_from:
			types_removals=[]
			if ind in multi_sent_amr_ids:
				types_removals.append('sentence_removal')
			if ind in amrs_with_ques_amr_ids:
				types_removals.append('question_removal')
			if ind in single_sent_arm_ids_with_arg:
				types_removals.append('op_plus_removal')
			if not types_removals:
				print('Warning: Decreasing Engagement was not applicable!!!')
				return
			type_removal=np.random.choice(types_removals)
			selected_utt_amr_ids = []
			if type_removal=="question_removal":
				self.question_removal_for_multi_sent_amrs(ind)
			elif type_removal=="sentence_removal":
				self.sentence_removal_for_multi_sent_amrs(ind)
			elif type_removal=="op_plus_removal":
				self.op_plus_removal_for_single_sent_amrs(ind)

	def direct_negate(self, ind, overall_perc=0.7):
		'''add polarity after the main concept of the identified utterane
		Param:
			ind: index of an utterance
			overall_perc: the perentage of manipulations
		'''
		amr=self.utts_amrs[ind]
		num_sents=self.get_num_sents(amr)
		amr_origin=amr
		amr=amr.split('\n')
		new_amr=''
		num_changes=0
		#we select overall_perc% of sentences in the utterance to be negated
		num_sents_choose= int(num_sents*overall_perc) if int(num_sents*overall_perc)>0 else 1
		for index, line in enumerate(amr):
			new_amr+=line+'\n'
			if re.findall('/ [a-z]+-[0-9][0-9]', line) and num_changes<num_sents_choose:
				#this is a main concept (verb) and needs to be negated
				indent=len(re.findall(r"^ *", line)[0])
				new_amr+=(indent+6)*' '+':polarity -\n'
				num_changes+=1
		return new_amr



	def indirect_negate(self, ind, overall_perc=0.7):
		'''extract the antonyms of the concepts in the amr and replace comncepts with them
		Param:
			ind: index of an utterance
			overall_perc: the perentage of manipulations
		'''
		curr_amr=self.utts_amrs[ind]
		new_amr=self.antonym_replacement(ind, 0.5)
		return new_amr



	def question_removal_for_multi_sent_amrs(self, id_selected):
		'''helper func specific to multi_sent_question_removal (engaging), it finds the location of the last sentence of question type and remove it
		Param:
			id_selected: index of selected conversation
		'''
		def get_question_lines(question_line_num, sentences_starting_line_nums, total_num_lines):
			#this function returns lines of amr in the conversation that contains the question (starting from its sentence)
			#if the question is in the last sent of amr return all the lines until the end of the amr
			total_sents = len(sentences_starting_line_nums)
			if question_line_num != -1 and total_sents > 0:
				last_starting_line_before_question = sentences_starting_line_nums[0]
				for curr_sent_idx, curr_sent_starting_line_num in enumerate(sentences_starting_line_nums):
					is_last_sent = (curr_sent_idx == total_sents - 1)
					if question_line_num >= curr_sent_starting_line_num:
						last_starting_line_before_question = curr_sent_starting_line_num
					else:
						break
				idx = sentences_starting_line_nums.index(last_starting_line_before_question)
				if idx < len(sentences_starting_line_nums) - 1: # not the last
					return list(range(last_starting_line_before_question, sentences_starting_line_nums[idx+1]))
				else:
					return list(range(last_starting_line_before_question, total_num_lines))
			return []

		# get lines number of the last question sentence
		curr_amr = self.utts_amrs[id_selected]
		curr_amr_lines = curr_amr.split('\n')
		question_line_num = self.locate_last_question_line_num_in_amr(curr_amr_lines)
		sentences_starting_line_nums = self.locate_sent_starting_line_nums_in_multi_sent_amr(curr_amr_lines)
		line_nums_to_remove = get_question_lines(question_line_num, sentences_starting_line_nums, len(curr_amr_lines))
		if len(line_nums_to_remove) > 0:
			self.remove_lines_from_amr(id_selected, curr_amr_lines, line_nums_to_remove, "remove question from multi-sent utterance")
		else:
			print("Problem occured in question_removal attempting to remove question from AMR:\n{}".format(self.utts_amrs[id_selected]))
			print("with amr question line (num: {}): {}".format(question_line_num, curr_amr_lines[question_line_num]))

	def sentence_removal_for_multi_sent_amrs(self, id_selected):
		'''helper func specific to multi_sent_sent_removal (engaging)
		Param:
			id_selected: index of selected conversation

		'''
		def get_lines_to_remove(sentences_starting_line_nums, total_num_lines):
			#this function returns the list including the stating line of the longest sentence in amr of the utterance till end
			total_sents = len(sentences_starting_line_nums)
			longest_sent_line_count = 0
			longest_sent_idx = None
			longest_sent_begin_at_line_num = None
			if total_sents > 0:
				for sent_idx, curr_sent_starting_line_num in enumerate(sentences_starting_line_nums):
					is_last_sent = (sent_idx == total_sents - 1)
					if not is_last_sent:
						sent_line_count = sentences_starting_line_nums[sent_idx + 1] - curr_sent_starting_line_num
					else:
						sent_line_count = total_num_lines - curr_sent_starting_line_num
					if sent_line_count > longest_sent_line_count:
						longest_sent_line_count = sent_line_count
						longest_sent_idx = sent_idx
						longest_sent_begin_at_line_num = curr_sent_starting_line_num
				if longest_sent_begin_at_line_num > 1:
					return list(range(longest_sent_begin_at_line_num, total_num_lines))
				else:
					return list(range(sentences_starting_line_nums[1], total_num_lines))
			return []
		# get lines numbers of the longest sentence and its following (if any)
		curr_amr = self.utts_amrs[id_selected]
		curr_amr_lines = curr_amr.split('\n')
		sentences_starting_line_nums = self.locate_sent_starting_line_nums_in_multi_sent_amr(curr_amr_lines)
		line_nums_to_remove = get_lines_to_remove(sentences_starting_line_nums, len(curr_amr_lines))
		if len(line_nums_to_remove) > 0:
			self.remove_lines_from_amr(id_selected, curr_amr_lines, line_nums_to_remove, "remove sentence from multi-sent utterance")
		else:
			print("Problem occured in sentence_removal attempting to remove sentence from AMR:\n{}".format(self.utts_amrs[id_selected]))

	def op_plus_removal_for_single_sent_amrs(self, id_selected, min_lines_to_remove=1, pattern_str_list=['op', 'ARG']):
		'''detect the complex substructure(s) in single sentence amrs and randomly choose to remove one and all its sublines
		Param:
			id_selected: index of selected conversation
			min_lines_to_remove: number of lines to remove
			pattern_str_list: type of operand to take into account
		'''
		def get_lines_to_remove(list_list_lines):
			# list_list_lines: list of list of lines to remove
			lines_to_remove=[]
			if len(list_list_lines)>1:
				select_ind = np.random.choice(len(list_list_lines))
				lines_to_remove = list_list_lines[select_ind]
			elif len(list_list_lines)==1:
				lines_to_remove = list_list_lines[0]
			return lines_to_remove

		# get lines numbers of the longest op[1-9]
		curr_amr = self.utts_amrs[id_selected]
		curr_amr_lines = curr_amr.split('\n')
		sub_structure_lines = self.get_single_snt_amr_sub_structures_for_specified_patterns(
			curr_amr_lines,
			pattern_str_list=pattern_str_list,
			min_line_required=min_lines_to_remove
		)
		line_nums_to_remove = get_lines_to_remove([val for val in sub_structure_lines.values()])
		if len(line_nums_to_remove) > 0:
			self.remove_lines_from_amr(id_selected, curr_amr_lines, line_nums_to_remove, "remove complex op/ARGs from single-sent utterance")
		else:
			print("Problem occured in op_plus_removal attempting to remove sentence from AMR:\n{}".format(self.utts_amrs[id_selected]))

	def get_single_snt_amr_sub_structures_for_specified_patterns(self, curr_amr_lines, pattern_str_list=['op', 'ARG'], min_line_required=2):
		'''return a dictionary of all op and ARGs in the amr and their sublines
		Param:
			id_selected: index of selected conversation
			curr_amr_lines: current utterance amr lines
			pattern_str_list: type of operand to take into account
			min_line_required: minimum lines required
		'''
		return_dict = {} # format: {starting line: list of lines of the structure (to remove)}
		for starting_line in range(len(curr_amr_lines)):
			curr_amr_line = curr_amr_lines[starting_line]
			includes_patterns = [True if re.match('[ ]*:{}[1-9]+ \('.format(pattern), curr_amr_line) else False for pattern in pattern_str_list]
			if any(includes_patterns):
				starting_line_indent_unit = len(curr_amr_line) - len(curr_amr_line.lstrip())
				for sub_structure_check_curr_at_line in range(starting_line+1, len(curr_amr_lines)):
					current_indent_unit = len(curr_amr_lines[sub_structure_check_curr_at_line]) \
							- len(curr_amr_lines[sub_structure_check_curr_at_line].lstrip())
					if current_indent_unit <= starting_line_indent_unit:
						if (sub_structure_check_curr_at_line - starting_line) >= min_line_required:
							return_dict[starting_line] = list(range(starting_line, sub_structure_check_curr_at_line)) # bug fixed, shouldn't add 1 to curr_at_line
						break
					else:
						# need to handle the case of last line!!
						if sub_structure_check_curr_at_line == len(curr_amr_lines) - 1: # last line, still less than (meaning all the rest is part of the sub-structure)
							return_dict[starting_line] = list(range(starting_line, sub_structure_check_curr_at_line+1))

		return return_dict



	def locate_last_question_line_num_in_amr(self, curr_amr_lines):
		'''return the last question in the current utterance amr
		Param:
			curr_amr_lines: current utterance amr lines
		'''
		return_line_num = -1
		for idx, line in enumerate(curr_amr_lines):
			if 'amr-unknown' in line:
				return_line_num = idx
		return return_line_num

	def locate_sent_starting_line_nums_in_multi_sent_amr(self, curr_amr_lines):
		'''return the line numbers in amr that include :snt
		Param:
			curr_amr_lines: current utterance amr lines
		'''
		line_nums = []
		for idx, line in enumerate(curr_amr_lines):
			if re.search(':snt[1-9]+', line):
				line_nums.append(idx)
		return line_nums

	def remove_lines_from_amr(self, amr_id, curr_amr_lines, line_nums_to_remove, method=""):
		'''remove all lines specified in line_nums_to_remove list from the current amr
		Param:
			amr_id: specified index
			curr_amr_lines: current utterance amr lines
			line_nums_to_remove: lines of amr to remove
			method: the type of removal
		'''
		remaining_lines = []
		for idx, line in enumerate(curr_amr_lines):
			if idx not in line_nums_to_remove:
				remaining_lines.append(line)
		amr_after_removal = '\n'.join(remaining_lines)

		if self.is_multisent(amr_after_removal):
			num_sent = self.get_num_sents(amr_after_removal)
			if num_sent == 1:
				remaining_lines = remaining_lines[1:]
				remaining_lines[0] = re.sub(':snt[1-9]+ ', '', remaining_lines[0])
				self.remove_extra_indentation(remaining_lines)
			elif num_sent > 1:
				remaining_lines_joined = "\n".join(remaining_lines)
				for i in range(num_sent):
					remaining_lines_joined = re.sub(
						':snt[1-9]+ ',
						':SNT{} '.format(str(i+1)),
						remaining_lines_joined,
						1
					)
				remaining_lines = re.sub(':SNT', ':snt', remaining_lines_joined).split('\n')

			elif num_sent < 1:
				remaining_lines = remaining_lines[1:]

		self.utts_amrs[amr_id] = '\n'.join(remaining_lines)




	def locate_ops_start_end_line_nums_in_single_amr(self, curr_amr_lines):
		'''return a list of tuples including each op's first and last lines indexes in the amr
		Param:
			curr_amr_lines: current utterance amr lines
		'''
		curr_line_idx = 0
		start_end_list = []
		while curr_line_idx < len(curr_amr_lines):
			curr_amr_line = curr_amr_lines[curr_line_idx]
			if re.match('[ ]*:op[1-9]+ \(', curr_amr_line):
				start = curr_line_idx
				end = start + self.get_num_sub_lines_including_start(curr_amr_lines, curr_line_idx)
				start_end_list.append((start, end))
				curr_line_idx = end
			else:
				curr_line_idx += 1
		return start_end_list



	def get_num_sub_lines_including_start(self, curr_amr_lines, curr_line_idx):
		'''return the number of sublines for a specific line (indicated by curr_line_idx) based on the indentation
		Param:
			curr_amr_lines: current utterance amr lines
			curr_line_idx: index of the current line 
		'''
		indent_base_unit = len(curr_amr_lines[curr_line_idx]) - len(curr_amr_lines[curr_line_idx].lstrip())
		line_count = 1
		for i in range(len(curr_amr_lines[curr_line_idx+1:])):
			curr_indent_unit = len(curr_amr_lines[i]) - len(curr_amr_lines[i].lstrip())
			if curr_indent_unit > indent_base_unit:
				line_count += 1
			else:
				break
		return line_count



	def synonym_replacement(self, amr_lines, overall_perc=0.8):
		'''replace overall_perc percentage of concepts or ops (that have synonyms) in the amr with their synonyms
		Param:
			amr_lines: current utterance amr lines
			overall_perc: perentage of manipulations
		'''
		assert(overall_perc <= 1)
		line_idxes_2_concepts_synonyms = {}

		for line_idx, line in enumerate(amr_lines):
			concept, synonym = None, None

			# verb
			if re.findall('/ [a-z]+-[0-9][0-9]', line):
				concept = line[:re.search('-[0-9][0-9]', line).start()]
				concept = concept.split('/ ')[1].strip()
				synonym = self.get_synonym(concept)
			# op noun
			elif re.findall(':op[0-9]+ \([a-z]+[0-9]* / [a-z]+[)]*', line):
				concept = line.split('/ ')[1].strip()
				while concept[-1]==')':
					concept = concept[:-1]
				synonym = self.get_synonym(concept)

			elif re.findall(':ARG[0-9]+ \([a-z]+[0-9]* / [a-z]+[)]*', line):
				concept = line.split('/ ')[1].strip()
				while concept[-1]==')':
					concept = concept[:-1]
				synonym = self.get_synonym(concept)

			elif re.findall(':mod \([a-z]+[0-9]* / [a-z]+[)]*', line):
				concept = line.split('/ ')[1].strip()
				while concept[-1]==')':
					concept = concept[:-1]
				synonym = self.get_synonym(concept)

			if synonym:
				line_idxes_2_concepts_synonyms[line_idx] = (concept, synonym)
		if not line_idxes_2_concepts_synonyms:
			return None
		num_replacements = max(int(overall_perc * len(line_idxes_2_concepts_synonyms)), 1)
		random_selected_idxes = np.random.choice(list(line_idxes_2_concepts_synonyms.keys()), size=num_replacements, replace=False)
		new_amr_lines = []
		for line_idx, line in enumerate(amr_lines):
			if line_idx not in random_selected_idxes:
				new_amr_lines.append(line)
			else:
				concept, synonym = line_idxes_2_concepts_synonyms[line_idx]
				new_amr_lines.append(re.sub(concept, synonym, line))

		return '\n'.join(new_amr_lines)



	def antonym_replacement(self, ind, overall_perc=0.5):
		'''replace overall_perc percentage of concepts or ops (that have antonyms) in the amr with their antonyms
		Param:
			ind: index of the conversarion
			overall_perc: perentage of manipulations
		'''
		assert(overall_perc <= 1)
		line_idxes_2_concepts_antonyms = {}
		amr=self.utts_amrs[ind]
		amr_lines=amr.split('\n')
		for line_idx, line in enumerate(amr_lines):
			concept, antonym = None, None

			# verb
			if re.findall('/ [a-z]+-[0-9][0-9]', line):
				concept = line[:re.search('-[0-9][0-9]', line).start()]
				concept = concept.split(' / ')[1].strip()
				antonym = self.get_antonym(concept)
			# op noun
			elif re.findall(':op[0-9]+ \([a-z]+[0-9]* / [a-z]+[)]*', line):
				concept = line.split(' / ')[1].strip()
				while concept[-1]==')':
					concept = concept[:-1]
				antonym = self.get_antonym(concept)

			elif re.findall(':ARG[0-9]+ \([a-z]+[0-9]* / [a-z]+[)]*', line):
				concept = line.split(' / ')[1].strip()
				while concept[-1]==')':
					concept = concept[:-1]
				antonym = self.get_antonym(concept)

			elif re.findall(':mod \([a-z]+[0-9]* / [a-z]+[)]*', line):
				concept = line.split(' / ')[1].strip()
				while concept[-1]==')':
					concept = concept[:-1]
				antonym = self.get_antonym(concept)

			if antonym:
				line_idxes_2_concepts_antonyms[line_idx] = (concept, antonym)
		num_replacements = max(int(overall_perc * len(line_idxes_2_concepts_antonyms)), 1)
		if not line_idxes_2_concepts_antonyms:
			new_amr_lines=self.direct_negate(ind)
			return new_amr_lines

		random_selected_idxes = np.random.choice(list(line_idxes_2_concepts_antonyms.keys()), size=num_replacements, replace=False)
		new_amr_lines = []
		for line_idx, line in enumerate(amr_lines):
			if line_idx not in random_selected_idxes:
				new_amr_lines.append(line)
			else:
				concept, antonym = line_idxes_2_concepts_antonyms[line_idx]
				new_amr_lines.append(re.sub(concept, antonym, line))

		return '\n'.join(new_amr_lines)


	def is_multisent(self, amr):
		'''check whether the utterance amr has multisentences
		Param:
			amr: amr of the conversation
		'''
		return True if len(re.findall(':snt[0-9]+', amr))>1    else False



	def includes_question(self, amr):
		'''check whether the utterance amr has questions
		Param:
			amr: amr of the conversation
		'''
		return True if 'amr-unknown' in amr else False

	
	def includes_arg(self, amr):
		'''check whether the utterance amr has ARGs
		Param:
			amr: amr of the conversation
		'''
		return True if len(re.findall('[ ]*ARG[1-9]+ \(', amr))>0 else False



	def add_indentation(self, amr, indent_level):
		'''add indentation to all levels of amr except the first line
		Param:
			amr: amr of the conversation
			indent_level: level of indent to apply
		'''
		amr_lines=amr.split('\n')
		new_amr=[amr_lines[0]]
		for amr_line in amr_lines[1:]:
			indent=len(amr_line) - len(amr_line.lstrip())
			new_amr.append((indent+(6*indent_level))*' '+amr_line.strip())
		return '\n'.join(new_amr)



	def increase_sents(self, amr, index):
		'''change sentences numbers of a multisentence amr starting from index
		Param:
			amr: amr of the conversation
			index: index of the sentence
		'''
		#
		parts=amr.split('\n')
		for ind, part in enumerate(parts):
			if ':snt' in part:
				parts[ind]=re.sub('snt[1-9]+', 'snt'+str(index), part)
				index+=1
		return '\n'.join(parts)

	def includes_at_least_2_ops(self, amr):
		'''check whether the amr has at least 2 ops
		Param:
			amr: amr of the conversation
		'''
		return self.get_num_ops_single_sent(amr) > 1



	def get_num_ops_single_sent(self, amr):
		'''return number of ops in the amr 
		Param:
			amr: amr of the conversation
		'''
		assert(not self.is_multisent(amr))
		return len(re.findall(':op[1-9]+ \(', amr))



	def get_num_sents(self, amr):
		'''return number of sentences in the amr 
		Param:
			amr: amr of the conversation
		'''
		num_sents=len(re.findall(':snt[1-9]+', amr))
		return num_sents if num_sents>0 else 1



	def remove_extra_indentation(self, amr_lines):
		'''remove extra indentations in the amr 
		Param:
			amr_lines: amr lines of the conversation
		'''
		indent_unit = 10000
		for i in range(len(amr_lines)):
			curr_left_space = len(amr_lines[i]) - len(amr_lines[i].lstrip())
			if curr_left_space < indent_unit:
				indent_unit = curr_left_space
		for i in range(len(amr_lines)):
			amr_lines[i] = amr_lines[i][indent_unit:]



	def get_pronouns(self, conv_amr):
		'''get the pronouns in conversation's amr and their position in the conversation (index of the utterances including those pronouns)
		Param:
			conv_amr: amr lines of the conversation
		'''
		pronouns_positions=defaultdict(list)
		for ind, utt_amr in enumerate(conv_amr):
			pronouns=re.findall('\/ (i|you|he|she|it|we|they)[\)]+' , utt_amr)
			for p in pronouns:
				pronouns_positions[p].append(ind)
		return pronouns_positions



	def get_nouns(self):
		'''return words with nouns types
		'''
		nouns=[]
		for utt in self.utts:
			words = nltk.word_tokenize(utt)
			tags = nltk.pos_tag(words)
			nouns+=[token for (token, tag) in tags if tag=='NN' or tag=='NNP']
		return nouns



	def get_utts_with_concepts(self):
		'''get the utterances in the conversation that have concepts alongside their index
		'''
		utt_with_concepts=[]
		utt_with_concepts_inds=[]
		for ind, utt_amr in enumerate(self.utts_amrs):
			if re.findall('/ [a-z]+-[0-9][0-9]', utt_amr):
				utt_with_concepts.append(utt_amr)
				utt_with_concepts_inds.append(ind)
		return utt_with_concepts, utt_with_concepts_inds



	def get_utt_items(self, utt):
		'''get the utterance's items such as args, ops and mods
		'''
		items={}
		items['concept']=set()
		items['arg']=set()
		items['op']=set()
		items['mod']=set()
		parts=utt.split('\n')
		for p in parts:
			concept=re.findall('/ [a-z]+-[0-9][0-9]', p)
			if concept:
				concept=concept[0]
				concept=concept[:re.search('-[0-9][0-9]', concept).start()]
				concept=concept.split('/ ')[1].strip()
				items['concept'].add(concept)
			elif re.findall(':ARG[0-9]+', p) and ' / ' in p:
				arg=p.split(' / ')[1].split(')')[0].split('\n')[0]
				items['arg'].add(arg)
			elif re.findall(':op[0-9]+', p) and ' / ' in p:
				op=p.split(' / ')[1].split(')')[0].split('\n')[0]
				items['op'].add(op)
			elif re.findall(':mod', p) and ' / ' in p:
				mod=p.split(' / ')[1].split(')')[0].split('\n')[0]
				items['mod'].add(mod)
		return items



	def get_all_items(self):
		'''get the concepts, args, ops and mods in all utts of the same conversation
		'''
		items={}
		items['concept']=[]
		items['arg']=[]
		items['op']=[]
		items['mod']=[]
		for ind, utt in enumerate(self.utts_amrs):
			utt_items= self.get_utt_items(utt)
			items['concept']+=utt_items['concept']
			items['arg']+=utt_items['arg']
			items['op']+=utt_items['op']
			items['mod']+=utt_items['mod']
		return items



if __name__=='__main__':
	parser = argparse.ArgumentParser(description='create coherence data')
	parser.add_argument('--data_path', type=str, default='data/topical_chat/', help='directory of input data')
	parser.add_argument('--o_data_path', type=str, default='data/topical_chat/', help='directory of output data')
	parser.add_argument('--fname', type=str, default='train', help='type of input data train/valid')
	args = parser.parse_args()
	fname=os.path.join(args.data_path,f'parsed_{args.fname}.txt')
	foutput=os.path.join(args.o_data_path,f'{args.fname}_amr_manamr_cont_coref_pirel_eng.txt')
	foutput_types=os.path.join(args.o_data_path,f'{args.fname}_amr_manamr_cont_coref_pirel_eng_mplt_types.txt')
	fw=open(foutput, 'w')
	fw_types=open(foutput_types, 'w')
	man_amr = Manipulation(fname, args, which_split=args.fname)
	
	type_manipulations=['contradiction', 'coreference_incosistency', 'partial_irrelevant', 'decrease_engagement']
	print(f'number of connversations to process is {man_amr.num_convs}')	

	for ind in range(man_amr.num_convs):
		man_amr.load_specific_utt_amrs(ind)
		sents, _ = gtos.generate(man_amr.utts_amrs)
		fw.write('</UTT>'.join(sents) + '</UTT>1\n')
		fw_types.write('no manipulations \n')	
		print('conv {}th'.format(ind))
		print('Manipulating')
		print('ORIGINAL UTTERANCES:')
		for ind, sent in enumerate(man_amr.utts):
			print(ind, sent)
		num_manipulations= np.random.choice(list(range(1,len(type_manipulations)+1)))
		manipulations=np.random.choice(type_manipulations, num_manipulations, replace=False)
		if 'coreference_incosistency' in manipulations:
			print('************************COREFERENCE_INCONSISTENCY******************************')
			man_amr.coreference(overall_perc=0.2)
			print('UPDATED UTTERANCES AFFTER COREFERENCE:')
			for ind, sent in enumerate(sents):
			 	print(ind, sent)
			 	print(man_amr.utts_amrs[ind])
		if 'contradiction' in manipulations:
			print('************************CONTRADICTION******************************')
			man_amr.contradiction(overall_perc=0.1)
			print('UPDATED UTTERANCES AFFTER CONTRADICTION:')
			for ind, sent in enumerate(sents):
			 	print(ind, sent)
			 	print(man_amr.utts_amrs[ind])
		if 'partial_irrelevant' in manipulations:
			print('************************PARTIAL_IRRELEVANT******************************')
			man_amr.partial_irrelevant(overall_perc=0.1, replace_chance=0.5)
			print('UPDATED UTTERANCES AFFTER PARTIAL IRRELAVANT:')
			for ind, sent in enumerate(sents):
			 	print(ind, sent)
			 	print(man_amr.utts_amrs[ind])
		if 'decrease_engagement' in manipulations:
			print('************************DECREASE_ENGAGEMENT******************************')
			man_amr.decrease_engagement(overall_perc = 0.2)
			sents, _ = gtos.generate(man_amr.utts_amrs)
			print('UPDATED UTTERANCES AFFTER DECREASING ENGAGEMENT:')
			for ind, sent in enumerate(sents):
			 	print(ind, sent)
			 	print(man_amr.utts_amrs[ind])
		print('conv manipulations is completed!')
		sents, _ = gtos.generate(man_amr.utts_amrs)
		for ind, sent in enumerate(sents):
			print(ind, sent)
		fw.write('</UTT>'.join(sents) + '</UTT>0\n')
		fw_types.write(','.join(manipulations) + '\n')
