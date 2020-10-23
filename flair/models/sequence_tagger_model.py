import warnings
import logging
from pathlib import Path

import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.autograd as autograd
import flair.nn
import torch

from flair.data import Dictionary, Sentence, Token, Label
from flair.datasets import DataLoader
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path

from typing import List, Tuple, Union

from flair.training_utils import Metric, Result, store_embeddings

from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import pdb
import copy

from ..variational_inference import MFVI
import time

log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


def to_scalar(var):
	return var.view(-1).detach().tolist()[0]


def argmax(vec):
	_, idx = torch.max(vec, 1)
	return to_scalar(idx)


def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
	_, idx = torch.max(vecs, 1)
	return idx


def log_sum_exp_batch(vecs):
	maxi = torch.max(vecs, 1)[0]
	maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
	recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
	# pdb.set_trace()
	return maxi + recti_

def log_sum_exp_vb(vec, m_size):
	"""
	calculate log of exp sum

	args:
		vec (batch_size, vanishing_dim, hidden_dim) : input tensor
		m_size : hidden_dim
	return:
		batch_size, hidden_dim
	"""
	_, idx = torch.max(vec, 1)  # B * 1 * M
	max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

	return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1,
																												m_size)  # B * M

def pad_tensors(tensor_list):
	ml = max([x.shape[0] for x in tensor_list])
	shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
	template = torch.zeros(*shape, dtype=torch.long, device=flair.device)
	lens_ = [x.shape[0] for x in tensor_list]
	for i, tensor in enumerate(tensor_list):
		template[i, : lens_[i]] = tensor

	return template, lens_


class SequenceTagger(flair.nn.Model):
	def __init__(
		self,
		hidden_size: int,
		embeddings: TokenEmbeddings,
		tag_dictionary: Dictionary,
		tag_type: str,
		use_crf: bool = True,
		use_mfvi: bool = False,
		use_rnn: bool = True,
		use_cnn: bool = False,
		rnn_layers: int = 1,
		dropout: float = 0.0,
		word_dropout: float = 0.05,
		locked_dropout: float = 0.5,
		train_initial_hidden_state: bool = False,
		pickle_module: str = "pickle",
		interpolation: float = 0.5,
		sentence_loss: bool = False,
		config = None,
		word_map = None,
		char_map = None,
		use_decoder_timer = True,
		debug = False,
		relearn_embeddings = False,
		map_embeddings = False,
		no_encoder = False,
		temperature: float = 1,
		relearn_size = -1,
		new_drop: bool = False,
		target_languages: int = 1,
	):
		"""
		Initializes a SequenceTagger
		:param hidden_size: number of hidden states in RNN
		:param embeddings: word embeddings used in tagger
		:param tag_dictionary: dictionary of tags you want to predict
		:param tag_type: string identifier for tag type
		:param use_crf: if True use CRF decoder, else project directly to tag space
		:param use_rnn: if True use RNN layer, otherwise use word embeddings directly
		:param rnn_layers: number of RNN layers
		:param dropout: dropout probability
		:param word_dropout: word dropout probability
		:param locked_dropout: locked dropout probability
		:param train_initial_hidden_state: if True, trains initial hidden state of RNN
		"""

		super(SequenceTagger, self).__init__()
		#add interpolation for target loss and distillation loss
		self.sentence_level_loss = sentence_loss
		self.interpolation = interpolation
		self.debug = debug
		self.use_rnn = use_rnn
		self.use_cnn = use_cnn
		self.hidden_size = hidden_size
		self.use_crf: bool = use_crf
		self.use_mfvi: bool = use_mfvi
		
		self.rnn_layers: int = rnn_layers
		self.target_languages: int = target_languages
		self.trained_epochs: int = 0
		self.embeddings = embeddings
		self.config = config
		self.use_decoder_timer = use_decoder_timer
		self.map_embeddings = map_embeddings
		# set the dictionaries
		self.tag_dictionary: Dictionary = tag_dictionary
		self.tag_type: str = tag_type
		self.tagset_size: int = len(tag_dictionary)
		self.relearn_size = relearn_size
		# initialize the network architecture
		self.nlayers: int = rnn_layers
		self.hidden_word = None
		self.new_drop = new_drop


		self.word_map = word_map
		self.char_map = char_map

		# dropouts
		self.use_dropout: float = dropout
		self.use_word_dropout: float = word_dropout
		self.use_locked_dropout: float = locked_dropout

		self.pickle_module = pickle_module

		# if dropout > 0.0:
		#   self.dropout = torch.nn.Dropout(dropout)

		# if word_dropout > 0.0:
		#   self.word_dropout = flair.nn.WordDropout(word_dropout)

		# if locked_dropout > 0.0:
		#   self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

		if not self.new_drop:
			if dropout > 0.0:
			  self.dropout = torch.nn.Dropout(dropout)

			if word_dropout > 0.0:
			  self.word_dropout = flair.nn.WordDropout(word_dropout)

			if locked_dropout > 0.0:
			  self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
		else:
			self.dropout1 = torch.nn.Dropout(p=dropout)
			self.dropout2 = torch.nn.Dropout(p=dropout)

		rnn_input_dim: int = self.embeddings.embedding_length
		if self.map_embeddings:
			rnn_input_dim=0
			self.map_linears = torch.nn.ModuleList()
			embedding_names={}
			for idx, embedding in enumerate(self.embeddings.embeddings):
				embedding_names[embedding.name]=idx
			for key in sorted(embedding_names.keys()):
				embedding = self.embeddings.embeddings[embedding_names[key]]
				if 'forward' in key or 'backward' in key:
					self.map_linears.append(torch.nn.Linear(embedding.embedding_length, 25))
					rnn_input_dim+=25
				else:
					self.map_linears.append(torch.nn.Linear(embedding.embedding_length, 50))
					rnn_input_dim+=50
		self.relearn_embeddings: bool = relearn_embeddings and not self.map_embeddings

		#==============debug==============
		# rnn_input_dim = rnn_input_dim*2
		#==============debug==============

		if self.relearn_embeddings:
			if relearn_size == -1:
				self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)
			else:
				self.embedding2nn = torch.nn.Linear(rnn_input_dim, relearn_size)
				rnn_input_dim = relearn_size

		self.train_initial_hidden_state = train_initial_hidden_state
		self.bidirectional = True
		self.rnn_type = "LSTM"
		if not self.use_rnn:
			self.bidirectional = False
		# bidirectional LSTM on top of embedding layer
		num_directions = 1
		if self.use_rnn:
			num_directions = 2 if self.bidirectional else 1

			if self.rnn_type in ["LSTM", "GRU"]:

				self.rnn = getattr(torch.nn, self.rnn_type)(
					rnn_input_dim,
					hidden_size,
					num_layers=self.nlayers,
					dropout=0.0 if self.nlayers == 1 else 0.5,
					bidirectional=True,
				)
				# Create initial hidden state and initialize it
				if self.train_initial_hidden_state:
					self.hs_initializer = torch.nn.init.xavier_normal_

					self.lstm_init_h = Parameter(
						torch.randn(self.nlayers * num_directions, self.hidden_size),
						requires_grad=True,
					)

					self.lstm_init_c = Parameter(
						torch.randn(self.nlayers * num_directions, self.hidden_size),
						requires_grad=True,
					)

					# TODO: Decide how to initialize the hidden state variables
					# self.hs_initializer(self.lstm_init_h)
					# self.hs_initializer(self.lstm_init_c)

			# final linear map to tag space
			self.linear = torch.nn.Linear(
				hidden_size * num_directions, len(tag_dictionary)
			)
		elif self.use_cnn:
			# cnn_layer=4
			# char_hidden_dim=50
			# hidden_dim=200
			# dropout=0.5
			# lstm_layer=1
			# bilstm=True
			# learning_rate=0.015
			# lr_decay=0.05
			# momentum=0
			# l2=1e-8
			# cnn_hidden = data.HP_hidden_dim
			self.word2cnn = torch.nn.Linear(rnn_input_dim, self.hidden_size)
			print("CNN layer: ", self.nlayers)
			self.cnn_list = torch.nn.ModuleList()
			self.cnn_drop_list = torch.nn.ModuleList()
			self.cnn_batchnorm_list = torch.nn.ModuleList()
			kernel = 3
			pad_size = int((kernel-1)/2)
			for idx in range(self.nlayers):
				self.cnn_list.append(torch.nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=kernel, padding=pad_size))
				self.cnn_drop_list.append(torch.nn.Dropout(0.5))
				self.cnn_batchnorm_list.append(torch.nn.BatchNorm1d(self.hidden_size))
			# final linear map to tag space
			self.linear = torch.nn.Linear(
				hidden_size * num_directions, len(tag_dictionary)
			)
		else:
			self.linear = torch.nn.Linear(
				self.embeddings.embedding_length, len(tag_dictionary)
			)

		if self.use_crf:
			self.transitions = torch.nn.Parameter(
				torch.randn(self.tagset_size, self.tagset_size)
			)
			self.transitions.detach()[
				self.tag_dictionary.get_idx_for_item(START_TAG), :
			] = -1e12
			self.transitions.detach()[
				:, self.tag_dictionary.get_idx_for_item(STOP_TAG)
			] = -1e12
		if self.use_mfvi:
			self.mfvi=MFVI(hidden_size*num_directions,self.tagset_size,**self.config['MFVI'])
		
		
		self.to(flair.device)
		
	def _get_state_dict(self):
		model_state = {
			"state_dict": self.state_dict(),
			"embeddings": self.embeddings,
			"hidden_size": self.hidden_size,
			"train_initial_hidden_state": self.train_initial_hidden_state,
			"tag_dictionary": self.tag_dictionary,
			"tag_type": self.tag_type,
			"use_crf": self.use_crf,
			"use_mfvi": self.use_mfvi,
			"target_languages": self.target_languages,
			"use_rnn": self.use_rnn,
			"use_cnn": self.use_cnn,
			"rnn_layers": self.rnn_layers,
			"use_word_dropout": self.use_word_dropout,
			"use_locked_dropout": self.use_locked_dropout,
			"word_map": self.word_map,
			"char_map": self.char_map,
			"config": self.config,
			"relearn_embeddings": self.relearn_embeddings,
			"map_embeddings": self.map_embeddings,
			"relearn_size": self.relearn_size,
			"new_drop": self.new_drop,
		}
		return model_state

	def _init_model_with_state_dict(state):

		use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
		use_word_dropout = (
			0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
		)
		use_locked_dropout = (
			0.0
			if not "use_locked_dropout" in state.keys()
			else state["use_locked_dropout"]
		)
		train_initial_hidden_state = (
			False
			if not "train_initial_hidden_state" in state.keys()
			else state["train_initial_hidden_state"]
		)
		new_drop = False if not "new_drop" in state.keys() else state["new_drop"]
		if new_drop:
			use_dropout=0.5
		use_cnn=state["use_cnn"] if 'use_cnn' in state else False
		model = SequenceTagger(
			hidden_size=state["hidden_size"],
			embeddings=state["embeddings"],
			tag_dictionary=state["tag_dictionary"],
			tag_type=state["tag_type"],
			use_crf=state["use_crf"],
			use_mfvi=state["use_mfvi"] if "use_mfvi" in state else False,
			target_languages=state["target_languages"] if "target_languages" in state else 1,
			use_rnn=state["use_rnn"],
			use_cnn=use_cnn,
			rnn_layers=state["rnn_layers"],
			dropout=use_dropout,
			word_dropout=use_word_dropout,
			locked_dropout=use_locked_dropout,
			word_map=None if 'word_map' not in state else state['word_map'],
			char_map=None if 'char_map' not in state else state['char_map'],
			train_initial_hidden_state=train_initial_hidden_state,
			config = state["config"] if "config" in state else None,
			relearn_embeddings = True if "relearn_embeddings" not in state else state["relearn_embeddings"],
			map_embeddings = False if "map_embeddings" not in state else state["map_embeddings"],
			relearn_size = -1 if "relearn_size" not in state else state["relearn_size"],
			new_drop = False if "new_drop" not in state else state["new_drop"],
		)
		model.load_state_dict(state["state_dict"])
		return model

	def evaluate(
		self,
		data_loader: DataLoader,
		out_path: Path = None,
		embeddings_storage_mode: str = "cpu",
		prediction_mode: bool = False
	) -> (Result, float):
		data_loader.assign_tags(self.tag_type,self.tag_dictionary)
		with torch.no_grad():
			eval_loss = 0

			batch_no: int = 0

			metric = Metric("Evaluation")

			lines: List[str] = []
			for batch in data_loader:
				batch_no += 1

				with torch.no_grad():
					features = self.forward(batch)

					loss = self._calculate_loss(features, batch)
					tags, _ = self._obtain_labels(features, batch)

				eval_loss += loss

				for (sentence, sent_tags) in zip(batch, tags):
					for (token, tag) in zip(sentence.tokens, sent_tags):
						token: Token = token
						token.add_tag_label("predicted", tag)

						# append both to file for evaluation
						eval_line = "{} {} {} {}\n".format(
							token.text,
							token.get_tag(self.tag_type).value,
							tag.value,
							tag.score,
						)
						lines.append(eval_line)
					lines.append("\n")
				for sentence in batch:
					# make list of gold tags
					gold_tags = [
						(tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)
					]
					# make list of predicted tags
					predicted_tags = [
						(tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
					]

					# check for true positives, false positives and false negatives
					for tag, prediction in predicted_tags:
						if (tag, prediction) in gold_tags:
							metric.add_tp(tag)
						else:
							metric.add_fp(tag)

					for tag, gold in gold_tags:
						if (tag, gold) not in predicted_tags:
							metric.add_fn(tag)
						else:
							metric.add_tn(tag)

				store_embeddings(batch, embeddings_storage_mode)

			eval_loss /= batch_no

			if out_path is not None:
				with open(out_path, "w", encoding="utf-8") as outfile:
					outfile.write("".join(lines))

			detailed_result = (
				f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
				f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
			)
			for class_name in metric.get_classes():
				detailed_result += (
					f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
					f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
					f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
					f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
					f"{metric.f_score(class_name):.4f}"
				)

			result = Result(
				main_score=metric.micro_avg_f_score(),
				log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
				log_header="PRECISION\tRECALL\tF1",
				detailed_results=detailed_result,
			)

			return result, eval_loss

	def write_prediction(
		self,
		data_loader: DataLoader,
		out_path: Path = None,
		embeddings_storage_mode: str = "cpu",
	) -> (Result, float):

		with torch.no_grad():
			eval_loss = 0

			batch_no: int = 0

			metric = Metric("Evaluation")

			lines: List[str] = []
			for batch in data_loader:
				batch_no += 1

				with torch.no_grad():
					features = self.forward(batch)
					tags, _ = self._obtain_labels(features, batch)


				for (sentence, sent_tags) in zip(batch, tags):
					previous_tag='O'
					for (token, tag) in zip(sentence.tokens, sent_tags):
						token: Token = token
						token.add_tag_label("predicted", tag)
						# append both to file for evaluation
						res=[]
						res.append(token.text)
						for label in token.tags:
							if self.tag_type==label:
								current_tag=tag.value.split('-')[-1]
								if current_tag=='O' or ('PER' not in current_tag and 'LOC' not in current_tag and 'MISC' not in current_tag and 'ORG' not in current_tag):
									write_value='O'
								elif previous_tag==current_tag:
									write_value='I-'+current_tag
								else:
									write_value='B-'+current_tag
								previous_tag=current_tag
								res.append(write_value)
							elif label=='predicted':
								pass
							else:
								res.append(token.tags[label].value)
						eval_line=' '.join(res)+'\n'
						# eval_line = "{} {} {} {}\n".format(
						#     token.text,
						#     token.get_tag(self.tag_type).value,
						#     tag.value,
						#     tag.score,
						# )
						lines.append(eval_line)
					lines.append("\n")

				store_embeddings(batch, embeddings_storage_mode)

			if out_path is not None:
				with open(out_path, "w", encoding="utf-8") as outfile:
					outfile.write("".join(lines))


	def forward_loss(
		self, data_points: Union[List[Sentence], Sentence], sort=True
	) -> torch.tensor:
		features = self.forward(data_points)
		return self._calculate_loss(features, data_points)


	def simple_forward_distillation_loss(
		self, data_points: Union[List[Sentence], Sentence], teacher_data_points: Union[List[Sentence], Sentence]=None, teacher=None, sort=True,
		interpolation=0.5, train_with_professor=False, professor_interpolation=0.5,
	) -> torch.tensor:
		lengths = [len(sentence.tokens) for sentence in data_points]
		longest_token_sequence_in_batch: int = max(lengths)
		features = self.forward(data_points)
		target_loss = self._calculate_loss(features, data_points)
		if self.distill_crf:
			teacher_tags=torch.stack([sentence.get_teacher_target() for sentence in data_points],0)
			# proprocess, convert k best to batch wise
			seq_len=teacher_tags.shape[1]
			best_k=teacher_tags.shape[-1]
			num_tags=features.shape[-1]
			tags=teacher_tags.transpose(1,2).reshape(-1,seq_len)
			features_input=features.unsqueeze(-1).repeat(1,1,1,best_k)
			features_input=features_input.permute(0,3,1,2).reshape(-1,seq_len,num_tags)
			lengths_input=torch.tensor(lengths)
			lengths_input=lengths_input.unsqueeze(-1).repeat(1,best_k)
			lengths_input=lengths_input.reshape(-1).cuda()

			distillation_loss=self._calculate_crf_distillation_loss(features_input,tags,lengths_input)
		else:
			teacher_features = torch.zeros(
				[
					len(data_points),
					longest_token_sequence_in_batch,
					features.shape[-1],
				],
				dtype=torch.float,
				device=flair.device,
			)
			for s_id, sentence in enumerate(data_points):
				# fill values with word embeddings
				if train_with_professor:
					teacher_features[s_id][: len(sentence)] = sentence.get_professor_teacher_prediction(professor_interpolation=professor_interpolation)
				else:
					teacher_features[s_id][: len(sentence)] = sentence.get_teacher_prediction()
			
			distillation_loss = self._calculate_distillation_loss(features, teacher_features, torch.Tensor(lengths), T=self.temperature, teacher_is_score=not self.distill_prob)
		return interpolation * distillation_loss + (1-interpolation) * target_loss
	def sequence_mask(self, lengths, max_len=None):
		"""
		Creates a boolean mask from sequence lengths.
		"""
		batch_size = lengths.numel()
		max_len = max_len or lengths.max()
		return (torch.arange(0, max_len)
				.type_as(lengths)
				.repeat(batch_size, 1)
				.lt(lengths.unsqueeze(1)))
	def _calculate_distillation_loss(self, features, teacher_features, lengths, T = 1):
		# TODO: time with mask, and whether this should do softmax
		# mask=self.sequence_mask(lengths, max_len).unsqueeze(-1).cuda().type_as(features)
		mask=self.mask
		KD_loss = torch.nn.functional.kl_div(F.log_softmax(features/T, dim=-1), F.softmax(teacher_features/T, dim=-1),reduction='none') * mask * T * T
		# KD_loss = KD_loss.sum()/mask.sum()
		if self.sentence_level_loss or self.use_crf:
			KD_loss = KD_loss.sum()/KD_loss.shape[0]
		else:
			KD_loss = KD_loss.sum()/mask.sum()
		return KD_loss
		# return torch.nn.functional.MSELoss(features, teacher_features, reduction='mean')

	def _calculate_crf_distillation_loss(self, features, tags, lengths):

		forward_score = self._forward_alg(features, lengths)
		gold_score = self._score_sentence(features, tags, lengths)
		score = forward_score - gold_score
		return score.mean()

	def predict(
		self,
		sentences: Union[List[Sentence], Sentence],
		mini_batch_size=32,
		embedding_storage_mode="none",
		all_tag_prob: bool = False,
		verbose=False,
		tag_name=None,
	) -> List[Sentence]:
		tag_type=self.tag_type if tag_name is None else tag_name
		with torch.no_grad():
			if isinstance(sentences, Sentence):
				sentences = [sentences]

			filtered_sentences = self._filter_empty_sentences(sentences)

			# remove previous embeddings
			store_embeddings(filtered_sentences, "none")

			# reverse sort all sequences by their length
			filtered_sentences.sort(key=lambda x: len(x), reverse=True)

			# make mini-batches
			batches = [
				filtered_sentences[x : x + mini_batch_size]
				for x in range(0, len(filtered_sentences), mini_batch_size)
			]

			# progress bar for verbosity
			if verbose:
				batches = tqdm(batches)

			for i, batch in enumerate(batches):

				if verbose:
					batches.set_description(f"Inferencing on batch {i}")

				with torch.no_grad():
					feature = self.forward(batch)
					tags, all_tags = self._obtain_labels(
						feature, batch, get_all_tags=all_tag_prob
					)

				for (sentence, sent_tags) in zip(batch, tags):
					for (token, tag) in zip(sentence.tokens, sent_tags):
						token.add_tag_label(tag_type, tag)

				# all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
				for (sentence, sent_all_tags) in zip(batch, all_tags):
					for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
						token.add_tags_proba_dist(tag_type, token_all_tags)

				# clearing token embeddings to save memory
				store_embeddings(batch, storage_mode=embedding_storage_mode)

			return sentences

	def forward(self, sentences: List[Sentence], prediction_mode = False):
		self.zero_grad()
		
		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

		longest_token_sequence_in_batch: int = max(lengths)
		
		self.embeddings.embed(sentences)
		if hasattr(sentences,'features'):
			if self.map_embeddings:
				new_list=[]
				# pdb.set_trace()
				for idx, embedding_name in enumerate(sorted(sentences.features.keys())):
					new_list.append(self.map_linears[idx](sentences.features[embedding_name].to(flair.device)))
				sentence_tensor = torch.cat(new_list,-1)
			sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())],-1)
			#==============debug==============
			# pdb.set_trace()
			# sentence_tensor = torch.cat([sentence_tensor,torch.zeros_like(sentence_tensor)],-1)
			# sentence_tensor = torch.cat([sentence_tensor*self.selection[0],torch.zeros_like(sentence_tensor)*self.selection[1]],-1)
			#==============debug==============

			if hasattr(self,'keep_embedding'):
				if self.map_embeddings:
					sentence_tensor=[]
					for idx, embedding_name in enumerate(sorted(sentences.features.keys())):
						sentence_tensor.append(self.map_linears[idx](sentences.features[embedding_name].to(flair.device)))
				else:
					sentence_tensor = [sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())]
				embedding_name = sorted(sentences.features.keys())[self.keep_embedding]

				if 'forward' in embedding_name or 'backward' in embedding_name:
					# sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys()) if 'forward' in x or 'backward' in x],-1)
					for idx, x in enumerate(sorted(sentences.features.keys())):
						if 'forward' not in x and 'backward' not in x:
							sentence_tensor[idx].fill_(0)
				else:
					for idx, x in enumerate(sorted(sentences.features.keys())):
						if x != embedding_name:
							sentence_tensor[idx].fill_(0)
				sentence_tensor = torch.cat(sentence_tensor,-1)
		else:
			# initialize zero-padded word embeddings tensor
			sentence_tensor = torch.zeros(
			  [
			      len(sentences),
			      longest_token_sequence_in_batch,
			      self.embeddings.embedding_length,
			  ],
			  dtype=torch.float,
			  device=flair.device,
			)

			for s_id, sentence in enumerate(sentences):
			  # fill values with word embeddings
			  sentence_tensor[s_id][: len(sentence)] = torch.cat(
			      [token.get_embedding().unsqueeze(0) for token in sentence], 0
			  )
			# sentence_tensor = sentence_tensor.to(flair.device)
		# # TODO: this can only be removed once the implementations of word_dropout and locked_dropout have a batch_first mode

		sentence_tensor = sentence_tensor.transpose_(0, 1)
		if self.new_drop:
		  sentence_tensor = self.dropout1(sentence_tensor)
		else:
			if self.use_dropout > 0.0:
				sentence_tensor = self.dropout(sentence_tensor)
			if self.use_word_dropout > 0.0:
				sentence_tensor = self.word_dropout(sentence_tensor)
			if self.use_locked_dropout > 0.0:
				sentence_tensor = self.locked_dropout(sentence_tensor)
		
		if self.relearn_embeddings:
			sentence_tensor = self.embedding2nn(sentence_tensor)

		if self.use_rnn:
			packed = torch.nn.utils.rnn.pack_padded_sequence(
				sentence_tensor, lengths, enforce_sorted=False
			)

			# if initial hidden state is trainable, use this state
			if self.train_initial_hidden_state:
				initial_hidden_state = [
					self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
					self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
				]
				rnn_output, hidden = self.rnn(packed, initial_hidden_state)
			else:
				rnn_output, hidden = self.rnn(packed)

			sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
				rnn_output, batch_first=True
			)
			if not self.new_drop:
				if self.use_dropout > 0.0:
				  sentence_tensor = self.dropout(sentence_tensor)
				# word dropout only before LSTM - TODO: more experimentation needed
				# if self.use_word_dropout > 0.0:
				#     sentence_tensor = self.word_dropout(sentence_tensor)
				if self.use_locked_dropout > 0.0:
				  sentence_tensor = self.locked_dropout(sentence_tensor)
			# if self.use_dropout > 0.0:
			#   sentence_tensor = self.dropout(sentence_tensor)
			# # word dropout only before LSTM - TODO: more experimentation needed
			# # if self.use_word_dropout > 0.0:
			# #     sentence_tensor = self.word_dropout(sentence_tensor)
			# if self.use_locked_dropout > 0.0:
			#   sentence_tensor = self.locked_dropout(sentence_tensor)
		elif self.use_cnn:
			
			# transpose to batch_first mode
			sentence_tensor = sentence_tensor.transpose_(0, 1)
			batch_size = len(sentences)
			word_in = torch.tanh(self.word2cnn(sentence_tensor)).transpose(2,1).contiguous()
			for idx in range(self.nlayers):
				if idx == 0:
					cnn_feature = F.relu(self.cnn_list[idx](word_in))
				else:
					cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
				cnn_feature = self.cnn_drop_list[idx](cnn_feature)
				if batch_size > 1:
					cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
			sentence_tensor = cnn_feature.transpose(2,1).contiguous()
		else:
			# transpose to batch_first mode
			sentence_tensor = sentence_tensor.transpose_(0, 1)
		if self.new_drop:
		  sentence_tensor = self.dropout2(sentence_tensor)
		if self.use_decoder_timer:
			self.time=time.time()
		features = self.linear(sentence_tensor)
		
		self.mask=self.sequence_mask(torch.tensor(lengths),longest_token_sequence_in_batch).cuda().type_as(features)
		if self.use_mfvi:
			# self.sent_feats=sentence_tensor
			token_feats=sentence_tensor
			unary_score=features
			
			features=self.mfvi(token_feats,unary_score,self.mask,lengths=torch.LongTensor(lengths).to(flair.device))
		
		return features

	def _score_sentence(self, feats, tags, lens_):

		start = torch.tensor(
			[self.tag_dictionary.get_idx_for_item(START_TAG)], device=flair.device
		)
		start = start[None, :].repeat(tags.shape[0], 1)

		stop = torch.tensor(
			[self.tag_dictionary.get_idx_for_item(STOP_TAG)], device=flair.device
		)
		stop = stop[None, :].repeat(tags.shape[0], 1)

		pad_start_tags = torch.cat([start, tags], 1)
		pad_stop_tags = torch.cat([tags, stop], 1)

		for i in range(len(lens_)):
			pad_stop_tags[i, lens_[i] :] = self.tag_dictionary.get_idx_for_item(
				STOP_TAG
			)

		score = torch.FloatTensor(feats.shape[0]).to(flair.device)

		for i in range(feats.shape[0]):
			r = torch.LongTensor(range(lens_[i])).to(flair.device)

			score[i] = torch.sum(
				self.transitions[
					pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
				]
			) + torch.sum(feats[i, r, tags[i, : lens_[i]]])

		return score

	def _calculate_loss(
		self, features: torch.tensor, sentences: List[Sentence]
	) -> float:

		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

		tag_list: List = []
		for s_id, sentence in enumerate(sentences):
			# get the tags in this sentence
			tag_idx: List[int] = [
				self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
				for token in sentence
			]
			# add tags as tensor
			tag = torch.tensor(tag_idx, device=flair.device)
			tag_list.append(tag)

		if self.use_crf:
			# pad tags if using batch-CRF decoder
			tags, _ = pad_tensors(tag_list)


			forward_score = self._forward_alg(features, lengths)
			gold_score = self._score_sentence(features, tags, lengths)

			score = forward_score - gold_score
			return score.mean()
		elif self.sentence_level_loss:
			score = 0
			for sentence_feats, sentence_tags, sentence_length in zip(
				features, tag_list, lengths
			):
				sentence_feats = sentence_feats[:sentence_length]

				score += torch.nn.functional.cross_entropy(
					sentence_feats, sentence_tags, reduction='sum'
				)
			score /= len(features)

			return score
		else:
			score = 0
			for sentence_feats, sentence_tags, sentence_length in zip(
				features, tag_list, lengths
			):
				sentence_feats = sentence_feats[:sentence_length]

				score += torch.nn.functional.cross_entropy(
					sentence_feats, sentence_tags
				)
			score /= len(features)
			return score

	def _obtain_labels(
		self, feature, sentences, get_all_tags: bool = False
	) -> (List[List[Label]], List[List[List[Label]]]):
		"""
		Returns a tuple of two lists:
		 - The first list corresponds to the most likely `Label` per token in each sentence.
		 - The second list contains a probability distribution over all `Labels` for each token
		   in a sentence for all sentences.
		"""

		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
		tags = []
		all_tags = []
		
		# if self.use_mfvi:
		#     token_feats=self.sent_feats
		#     unary_score=features
		#     
		#     q_value=self.mfvi(token_feats,unary_score,mask)
		#     q_value=self.mfvi()
		if not self.use_crf:
			distribution = F.softmax(feature, dim=-1)
			_, indices = torch.max(feature, -1)
			sentrange=torch.arange(0,distribution.shape[1]).long().cuda()

		for i, vals in enumerate(zip(feature, lengths)):
			feats, length=vals
			if self.use_crf:
				confidences, tag_seq, scores = self._viterbi_decode(
					feats[:length], all_scores=get_all_tags, current_idx = i,
				)
			else:
				tag_seq = []
				confidences = []
				scores = []
				tag_seq = indices[i][:length].tolist()
				confidences = distribution[i][sentrange,indices[i]][:length].tolist()
				scores = distribution[i][:length].tolist()
				# for backscore in feats[:length]:
				#     softmax = F.softmax(backscore, dim=0)
				#     _, idx = torch.max(backscore, 0)
				#     prediction = idx.item()
				#     tag_seq.append(prediction)
				#     confidences.append(softmax[prediction].item())
				#     scores.append(softmax.tolist())
				# if new_tag_seq!=tag_seq or new_confidences!=confidences or new_scores!=scores:

			tags.append(
				[
					Label(self.tag_dictionary.get_item_for_index(tag), conf)
					for conf, tag in zip(confidences, tag_seq)
				]
			)

			if get_all_tags:
				all_tags.append(
					[
						[
							Label(
								self.tag_dictionary.get_item_for_index(score_id), score
							)
							for score_id, score in enumerate(score_dist)
						]
						for score_dist in scores
					]
				)

		return tags, all_tags

	def _obtain_labels_speed(
		self, feature, sentences, get_all_tags: bool = False
	) -> (List[List[Label]], List[List[List[Label]]]):
		"""
		Remove the tag assignment in the speed comparison
		"""

		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
		tags = []
		all_tags = []
		
		# if self.use_mfvi:
		#     token_feats=self.sent_feats
		#     unary_score=features
		#     
		#     q_value=self.mfvi(token_feats,unary_score,mask)
		#     q_value=self.mfvi()
		if not self.use_crf:
			distribution = F.softmax(feature, dim=-1)
			_, indices = torch.max(feature, -1)
			sentrange=torch.arange(0,distribution.shape[1]).long().cuda()
		else:
			for i, vals in enumerate(zip(feature, lengths)):
				feats, length=vals
				if self.use_crf:
					confidences, tag_seq, scores = self._viterbi_decode(
						feats[:length], all_scores=get_all_tags, current_idx = i,
					)

		return None

	def _viterbi_decode(self, feats, all_scores: bool = False, current_idx = 0):
		backpointers = []
		backscores = []

		init_vvars = (
			torch.FloatTensor(1, self.tagset_size).to(flair.device).fill_(-1e12)
		)
		init_vvars[0][self.tag_dictionary.get_idx_for_item(START_TAG)] = 0
		forward_var = init_vvars
		for i, feat in enumerate(feats):
			next_tag_var = (
				forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size)
				+ self.transitions
			)
			_, bptrs_t = torch.max(next_tag_var, dim=1)
			viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
			forward_var = viterbivars_t + feat
			backscores.append(forward_var)
			backpointers.append(bptrs_t)
		terminal_var = (
			forward_var
			+ self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)]
		)
		terminal_var.detach()[self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -1e12
		terminal_var.detach()[
			self.tag_dictionary.get_idx_for_item(START_TAG)
		] = -1e12
		best_tag_id = argmax(terminal_var.unsqueeze(0))

		best_path = [best_tag_id]

		for bptrs_t in reversed(backpointers):
			best_tag_id = bptrs_t[best_tag_id]
			best_path.append(best_tag_id)

		best_scores = []
		for backscore in backscores:
			softmax = F.softmax(backscore, dim=0)
			_, idx = torch.max(backscore, 0)
			prediction = idx.item()
			best_scores.append(softmax[prediction].item())

		start = best_path.pop()
		assert start == self.tag_dictionary.get_idx_for_item(START_TAG)
		best_path.reverse()

		scores = []
		# return all scores if so selected
		if all_scores:
			for backscore in backscores:
				softmax = F.softmax(backscore, dim=0)
				scores.append([elem.item() for elem in softmax.flatten()])

			for index, (tag_id, tag_scores) in enumerate(zip(best_path, scores)):
				if type(tag_id) != int and tag_id.item() != np.argmax(tag_scores):
					swap_index_score = np.argmax(tag_scores)
					scores[index][tag_id.item()], scores[index][swap_index_score] = (
						scores[index][swap_index_score],
						scores[index][tag_id.item()],
					)
				elif type(tag_id) == int and tag_id != np.argmax(tag_scores):
					swap_index_score = np.argmax(tag_scores)
					scores[index][tag_id], scores[index][swap_index_score] = (
						scores[index][swap_index_score],
						scores[index][tag_id],
					)

		return best_scores, best_path, scores

	def _forward_alg(self, feats, lens_, distill_mode=False, T = 1):

		init_alphas = torch.FloatTensor(self.tagset_size).fill_(-1e12)
		init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0
		forward_var = torch.zeros(
			feats.shape[0],
			feats.shape[1] + 1,
			feats.shape[2],
			dtype=torch.float,
			device=flair.device,
		)
		forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)
		
		transitions = self.transitions.view(
			1, self.transitions.shape[0], self.transitions.shape[1]
		).repeat(feats.shape[0], 1, 1)
		if T!=1:
			transitions = transitions/T
			feats = feats/T
		for i in range(feats.shape[1]):

			emit_score = feats[:, i, :]

			tag_var = (
				emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
				+ transitions
				+ forward_var[:, i, :][:, :, None]
				.repeat(1, 1, transitions.shape[2])
				.transpose(2, 1)
			)

			max_tag_var, _ = torch.max(tag_var, dim=2)

			tag_var = tag_var - max_tag_var[:, :, None].repeat(
				1, 1, transitions.shape[2]
			)

			agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

			cloned = forward_var.clone()
			cloned[:, i + 1, :] = max_tag_var + agg_

			forward_var = cloned
		
		if distill_mode:
			# from the first tag to the last tag
			# forward_var = forward_var[:,1:].clone()
			return forward_var[:,1:]

		forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
		terminal_var = forward_var + self.transitions[
			self.tag_dictionary.get_idx_for_item(STOP_TAG)
		][None, :].repeat(forward_var.shape[0], 1)/T

		alpha = log_sum_exp_batch(terminal_var)

		return alpha


	def _backward_alg(self, feats, lens_, T = 1, distill_mode=True):
		# reverse the transitions
		bw_transitions=self.transitions.transpose(0,1)
		# n * m * d
		reversed_feats = torch.zeros_like(feats)
		
		for i, feat in enumerate(feats):
			# m * d -> k * d, reverse over tokens -> m * d
			reversed_feats[i][:lens_[i]] = feat[:lens_[i]].flip([0])
			# reverse_feats[i][:lens_[i]] = feat[:lens_[i]].filp(0)
		
		init_alphas = torch.FloatTensor(self.tagset_size).fill_(-1e12)
		init_alphas[self.tag_dictionary.get_idx_for_item(STOP_TAG)] = 0.0
		
		forward_var = torch.zeros(
			reversed_feats.shape[0],
			reversed_feats.shape[1] + 1,
			reversed_feats.shape[2],
			dtype=torch.float,
			device=flair.device,
		)
		forward_var[:, 0, :] = init_alphas[None, :].repeat(reversed_feats.shape[0], 1)
		transitions = bw_transitions.view(
			1, bw_transitions.shape[0], bw_transitions.shape[1]
		).repeat(reversed_feats.shape[0], 1, 1)

		if T!=1:
			transitions = transitions/T
			reversed_feats = reversed_feats/T

		for i in range(reversed_feats.shape[1]):
			if i == 0:
				emit_score = torch.zeros_like(reversed_feats[:, 0, :])
			else:
				emit_score = reversed_feats[:, i-1, :]
			# pdb.set_trace()
			tag_var = (
				emit_score[:, None, :].repeat(1, transitions.shape[2], 1)
				+ transitions
				+ forward_var[:, i, :][:, :, None]
				.repeat(1, 1, transitions.shape[2])
				.transpose(2, 1)
			)

			max_tag_var, _ = torch.max(tag_var, dim=2)

			tag_var = tag_var - max_tag_var[:, :, None].repeat(
				1, 1, transitions.shape[2]
			)

			agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

			cloned = forward_var.clone()
			cloned[:, i + 1, :] = max_tag_var + agg_

			forward_var = cloned
		# if self.distill_posterior:
		if distill_mode:
			backward_var = forward_var[:,1:].clone()
			new_backward_var = torch.zeros_like(backward_var)
			for i, var in enumerate(backward_var):
				
				# flip over tokens, [num_tokens * num_tags]
				new_backward_var[i,:lens_[i]] = var[:lens_[i]].flip([0])
				
			return new_backward_var

		forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
		terminal_var = forward_var + bw_transitions[
			self.tag_dictionary.get_idx_for_item(START_TAG)
		][None, :].repeat(forward_var.shape[0], 1) + reversed_feats[range(reversed_feats.shape[0]), lens_-1, :]/T

		alpha = log_sum_exp_batch(terminal_var)

		return alpha

	@staticmethod
	def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
		filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
		if len(sentences) != len(filtered_sentences):
			log.warning(
				"Ignore {} sentence(s) with no tokens.".format(
					len(sentences) - len(filtered_sentences)
				)
			)
		return filtered_sentences

	def _fetch_model(model_name) -> str:

		model_map = {}

		aws_resource_path_v04 = (
			"https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4"
		)

		model_map["ner"] = "/".join(
			[aws_resource_path_v04, "NER-conll03-english", "en-ner-conll03-v0.4.pt"]
		)

		model_map["ner-fast"] = "/".join(
			[
				aws_resource_path_v04,
				"NER-conll03--h256-l1-b32-p3-0.5-%2Bglove%2Bnews-forward-fast%2Bnews-backward-fast-normal-locked0.5-word0.05--release_4",
				"en-ner-fast-conll03-v0.4.pt",
			]
		)

		model_map["ner-ontonotes"] = "/".join(
			[
				aws_resource_path_v04,
				"release-ner-ontonotes-0",
				"en-ner-ontonotes-v0.4.pt",
			]
		)

		model_map["ner-ontonotes-fast"] = "/".join(
			[
				aws_resource_path_v04,
				"release-ner-ontonotes-fast-0",
				"en-ner-ontonotes-fast-v0.4.pt",
			]
		)

		for key in ["ner-multi", "multi-ner"]:
			model_map[key] = "/".join(
				[
					aws_resource_path_v04,
					"release-quadner-512-l2-multi-embed",
					"quadner-large.pt",
				]
			)

		for key in ["ner-multi-fast", "multi-ner-fast"]:
			model_map[key] = "/".join(
				[aws_resource_path_v04, "NER-multi-fast", "ner-multi-fast.pt"]
			)

		for key in ["ner-multi-fast-learn", "multi-ner-fast-learn"]:
			model_map[key] = "/".join(
				[
					aws_resource_path_v04,
					"NER-multi-fast-evolve",
					"ner-multi-fast-learn.pt",
				]
			)

		model_map["pos"] = "/".join(
			[
				aws_resource_path_v04,
				"POS-ontonotes--h256-l1-b32-p3-0.5-%2Bglove%2Bnews-forward%2Bnews-backward-normal-locked0.5-word0.05--v0.4_0",
				"en-pos-ontonotes-v0.4.pt",
			]
		)

		model_map["pos-fast"] = "/".join(
			[
				aws_resource_path_v04,
				"release-pos-fast-0",
				"en-pos-ontonotes-fast-v0.4.pt",
			]
		)

		for key in ["pos-multi", "multi-pos"]:
			model_map[key] = "/".join(
				[
					aws_resource_path_v04,
					"release-dodekapos-512-l2-multi",
					"pos-multi-v0.1.pt",
				]
			)

		for key in ["pos-multi-fast", "multi-pos-fast"]:
			model_map[key] = "/".join(
				[aws_resource_path_v04, "UPOS-multi-fast", "pos-multi-fast.pt"]
			)

		model_map["frame"] = "/".join(
			[aws_resource_path_v04, "release-frame-1", "en-frame-ontonotes-v0.4.pt"]
		)

		model_map["frame-fast"] = "/".join(
			[
				aws_resource_path_v04,
				"release-frame-fast-0",
				"en-frame-ontonotes-fast-v0.4.pt",
			]
		)

		model_map["chunk"] = "/".join(
			[
				aws_resource_path_v04,
				"NP-conll2000--h256-l1-b32-p3-0.5-%2Bnews-forward%2Bnews-backward-normal-locked0.5-word0.05--v0.4_0",
				"en-chunk-conll2000-v0.4.pt",
			]
		)

		model_map["chunk-fast"] = "/".join(
			[
				aws_resource_path_v04,
				"release-chunk-fast-0",
				"en-chunk-conll2000-fast-v0.4.pt",
			]
		)

		model_map["de-pos"] = "/".join(
			[aws_resource_path_v04, "release-de-pos-0", "de-pos-ud-hdt-v0.4.pt"]
		)

		model_map["de-pos-fine-grained"] = "/".join(
			[
				aws_resource_path_v04,
				"POS-fine-grained-german-tweets",
				"de-pos-twitter-v0.1.pt",
			]
		)

		model_map["de-ner"] = "/".join(
			[aws_resource_path_v04, "release-de-ner-0", "de-ner-conll03-v0.4.pt"]
		)

		model_map["de-ner-germeval"] = "/".join(
			[aws_resource_path_v04, "NER-germeval", "de-ner-germeval-0.4.1.pt"]
		)

		model_map["fr-ner"] = "/".join(
			[aws_resource_path_v04, "release-fr-ner-0", "fr-ner-wikiner-0.4.pt"]
		)
		model_map["nl-ner"] = "/".join(
			[aws_resource_path_v04, "NER-conll2002-dutch", "nl-ner-conll02-v0.1.pt"]
		)

		cache_dir = Path("models")
		if model_name in model_map:
			model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

		return model_name

	def get_transition_matrix(self):
		data = []
		for to_idx, row in enumerate(self.transitions):
			for from_idx, column in enumerate(row):
				row = [
					self.tag_dictionary.get_item_for_index(from_idx),
					self.tag_dictionary.get_item_for_index(to_idx),
					column.item(),
				]
				data.append(row)
			data.append(["----"])
		print(tabulate(data, headers=["FROM", "TO", "SCORE"]))
	def _viterbi_decode_nbest(self, feats, mask, nbest):
		"""
		Code from NCRFpp with some modification: https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py
		"""
		"""
			input:
				feats: (batch, seq_len, self.tag_size+2)
				mask: (batch, seq_len)
			output:
				decode_idx: (batch, nbest, seq_len) decoded sequence
				path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
				nbest decode for sentence with one token is not well supported, to be optimized
		"""

		batch_size = feats.size(0)
		seq_len = feats.size(1)
		tag_size = feats.size(2)
		assert(tag_size == self.tagset_size)
		## calculate sentence length for each sentence
		length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long()
		## mask to (seq_len, batch_size)
		mask = mask.transpose(1,0).contiguous()
		ins_num = seq_len * batch_size
		## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
		feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
		## need to consider start
		
		scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
		scores = scores.view(seq_len, batch_size, tag_size, tag_size)

		# build iter
		seq_iter = enumerate(scores)
		## record the position of best score
		back_points = list()
		partition_history = list()
		##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
		# mask = 1 + (-1)*mask
		mask =  (1 - mask.long()).byte()
		mask=mask.bool()
		_, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
		# only need start from start_tag
		partition = inivalues[:, self.tag_dictionary.get_idx_for_item(START_TAG), :].clone()  # bat_size * to_target_size
		## initial partition [batch_size, tag_size]
		partition_history.append(partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest))
		# iter over last scores
		for idx, cur_values in seq_iter:
			if idx == 1:
				cur_values = cur_values.view(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
			else:
				# previous to_target is current from_target
				# partition: previous results log(exp(from_target)), #(batch_size * nbest * from_target)
				# cur_values: batch_size * from_target * to_target
				cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size) + partition.contiguous().view(batch_size, tag_size, nbest, 1).expand(batch_size, tag_size, nbest, tag_size)
				## compare all nbest and all from target
				cur_values = cur_values.view(batch_size, tag_size*nbest, tag_size)
				# print "cur size:",cur_values.size()
			partition, cur_bp = torch.topk(cur_values, nbest, 1)
			## cur_bp/partition: [batch_size, nbest, tag_size], id should be normize through nbest in following backtrace step
			# print partition[:,0,:]
			# print cur_bp[:,0,:]
			# print "nbest, ",idx
			if idx == 1:
				cur_bp = cur_bp*nbest
			partition = partition.transpose(2,1)
			cur_bp = cur_bp.transpose(2,1)

			# print partition
			# exit(0)
			#partition: (batch_size * to_target * nbest)
			#cur_bp: (batch_size * to_target * nbest) Notice the cur_bp number is the whole position of tag_size*nbest, need to convert when decode
			partition_history.append(partition)
			## cur_bp: (batch_size,nbest, tag_size) topn source score position in current tag
			## set padded label as 0, which will be filtered in post processing
			## mask[idx] ? mask[idx-1]
			cur_bp.masked_fill_(mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0)
			# print cur_bp[0]
			back_points.append(cur_bp)
		### add score to final STOP_TAG
		partition_history = torch.cat(partition_history,0).view(seq_len, batch_size, tag_size, nbest).transpose(1,0).contiguous() ## (batch_size, seq_len, nbest, tag_size)
		### get the last position for each setences, and select the last partitions using gather()
		last_position = length_mask.view(batch_size,1,1,1).expand(batch_size, 1, tag_size, nbest) - 1
		last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, nbest, 1)
		### calculate the score from last partition to end state (and then select the STOP_TAG from it)
		last_values = last_partition.expand(batch_size, tag_size, nbest, tag_size) + self.transitions.view(1, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size)
		last_values = last_values.view(batch_size, tag_size*nbest, tag_size)
		end_partition, end_bp = torch.topk(last_values, nbest, 1)
		## end_partition: (batch, nbest, tag_size)
		end_bp = end_bp.transpose(2,1)
		# end_bp: (batch, tag_size, nbest)
		pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size, nbest)).long().to(device=flair.device)
		back_points.append(pad_zero)
		back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

		## select end ids in STOP_TAG
		pointer = end_bp[:, self.tag_dictionary.get_idx_for_item(STOP_TAG), :] ## (batch_size, nbest)
		insert_last = pointer.contiguous().view(batch_size, 1, 1, nbest).expand(batch_size, 1, tag_size, nbest)
		back_points = back_points.transpose(1,0).contiguous()
		## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
		# print "lp:",last_position
		# print "il:",insert_last[0]
		# exit(0)
		## copy the ids of last position:insert_last to back_points, though the last_position index
		## last_position includes the length of batch sentences
		# print "old:", back_points[9,0,:,:]
		back_points.scatter_(1, last_position, insert_last)
		## back_points: [batch_size, seq_length, tag_size, nbest]
		# print "new:", back_points[9,0,:,:]
		# exit(0)
		# print pointer[2]
		'''
		back_points: in simple demonstratration
		x,x,x,x,x,x,x,x,x,7
		x,x,x,x,x,4,0,0,0,0
		x,x,6,0,0,0,0,0,0,0
		'''

		back_points = back_points.transpose(1,0).contiguous()
		# print back_points[0]
		## back_points: (seq_len, batch, tag_size, nbest)
		## decode from the end, padded position ids are 0, which will be filtered in following evaluation
		decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size, nbest)).to(device=flair.device)
		
		decode_idx[-1] = pointer.data/nbest
		# print "pointer-1:",pointer[2]
		# exit(0)
		# use old mask, let 0 means has token
		for idx in range(len(back_points)-2, -1, -1):
			# print "pointer: ",idx,  pointer[3]
			# print "back:",back_points[idx][3]
			# print "mask:",mask[idx+1,3]
			new_pointer = torch.gather(back_points[idx].view(batch_size, tag_size*nbest), 1, pointer.contiguous().view(batch_size,nbest))
			decode_idx[idx] = new_pointer.data/nbest
			# # use new pointer to remember the last end nbest ids for non longest
			pointer = new_pointer + pointer.contiguous().view(batch_size,nbest)*mask[idx].view(batch_size,1).expand(batch_size, nbest).long()

		# exit(0)
		path_score = None
		decode_idx = decode_idx.transpose(1,0)
		## decode_idx: [batch, seq_len, nbest]
		# print decode_idx[:,:,0]
		# print "nbest:",nbest
		# print "diff:", decode_idx[:,:,0]- decode_idx[:,:,4]
		# print decode_idx[:,0,:]
		# exit(0)

		### calculate probability for each sequence
		scores = end_partition[:, :, self.tag_dictionary.get_idx_for_item(STOP_TAG)]
		## scores: [batch_size, nbest]
		max_scores,_ = torch.max(scores, 1)
		minus_scores = scores - max_scores.view(batch_size,1).expand(batch_size, nbest)
		path_score = F.softmax(minus_scores, 1)
		## path_score: [batch_size, nbest]
		# exit(0)
		return path_score, decode_idx



class FastSequenceTagger(SequenceTagger):
	def _init_model_with_state_dict(state):
		use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
		use_word_dropout = (
			0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
		)
		use_locked_dropout = (
			0.0
			if not "use_locked_dropout" in state.keys()
			else state["use_locked_dropout"]
		)
		train_initial_hidden_state = (
			False
			if not "train_initial_hidden_state" in state.keys()
			else state["train_initial_hidden_state"]
		)
		use_cnn=state["use_cnn"] if 'use_cnn' in state else False
		model = FastSequenceTagger(
			hidden_size=state["hidden_size"],
			embeddings=state["embeddings"],
			tag_dictionary=state["tag_dictionary"],
			tag_type=state["tag_type"],
			use_crf=state["use_crf"],
			use_mfvi=state["use_mfvi"] if "use_mfvi" in state else False,
			target_languages=state["target_languages"] if "target_languages" in state else 1,
			use_rnn=state["use_rnn"],
			use_cnn=use_cnn,
			rnn_layers=state["rnn_layers"],
			dropout=use_dropout,
			word_dropout=use_word_dropout,
			locked_dropout=use_locked_dropout,
			train_initial_hidden_state=train_initial_hidden_state,
			word_map=None if 'word_map' not in state else state['word_map'],
			char_map=None if 'char_map' not in state else state['char_map'],
			config=state['config'] if "config" in state else None,
			relearn_embeddings = True if "relearn_embeddings" not in state else state["relearn_embeddings"],
			map_embeddings = False if "map_embeddings" not in state else state["map_embeddings"],
			relearn_size = -1 if "relearn_size" not in state else state["relearn_size"],
			new_drop = False if "new_drop" not in state else state["new_drop"],
		)
		model.load_state_dict(state["state_dict"])
		return model

	def forward_loss(
		self, data_points: Union[List[Sentence], Sentence], sort=True
	) -> torch.tensor:
		features = self.forward(data_points)
		# lengths = [len(sentence.tokens) for sentence in data_points]
		# longest_token_sequence_in_batch: int = max(lengths)

		# max_len = features.shape[1]
		# mask=self.sequence_mask(torch.tensor(lengths), max_len).cuda().type_as(features)
		loss = self._calculate_loss(features, data_points, self.mask)
		return loss


	def sequence_mask(self, lengths, max_len=None):
		"""
		Creates a boolean mask from sequence lengths.
		"""
		batch_size = lengths.numel()
		max_len = max_len or lengths.max()
		return (torch.arange(0, max_len)
				.type_as(lengths)
				.repeat(batch_size, 1)
				.lt(lengths.unsqueeze(1)))

	def _calculate_loss(
		self, features: torch.tensor, sentences: List[Sentence], mask: torch.tensor,
	) -> float:

		lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

		tag_list: List = []
		try:
			tag_list=torch.stack([getattr(sentence,self.tag_type+'_tags').to(flair.device) for sentence in sentences],0).long()
		except:
			tag_list: List = []
			for s_id, sentence in enumerate(sentences):
				# get the tags in this sentence
				tag_idx: List[int] = [
					self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
					for token in sentence
				]
				# add tags as tensor
				tag = torch.tensor(tag_idx, device=flair.device)
				tag_list.append(tag)

			tag_list, _ = pad_tensors(tag_list)
		
		if self.use_crf:
			forward_score = self._forward_alg(features, lengths)
			gold_score = self._score_sentence(features, tag_list, torch.tensor(lengths), mask=mask)
			score = forward_score - gold_score
			score = score.mean()
		# elif self.use_mfvi:
		#     token_feats=self.sent_feats
		#     unary_score=features
		#     
		#     q_value=self.mfvi(token_feats,unary_score,mask)
		#     score = torch.nn.functional.cross_entropy(q_value.view(-1,q_value.shape[-1]), tag_list.view(-1,), reduction='none') * mask.view(-1,)
		#     if self.sentence_level_loss:
		#         score = score.sum()/features.shape[0]
		#     else:
		#         score = score.sum()/mask.sum()
		else:
			score = torch.nn.functional.cross_entropy(features.view(-1,features.shape[-1]), tag_list.view(-1,), reduction='none') * mask.view(-1,)
			if self.sentence_level_loss or self.use_crf:
				score = score.sum()/features.shape[0]
			else:
				score = score.sum()/mask.sum()
		return score

	def entropy_loss(self,distribution):
		return distribution.softmax(-1)*distribution.log_softmax(-1)
	def _score_sentence(self, feats, tags, lens_,mask=None):
		start = torch.tensor(
			[self.tag_dictionary.get_idx_for_item(START_TAG)], device=flair.device
		)
		start = start[None, :].repeat(tags.shape[0], 1)

		stop = torch.tensor(
			[self.tag_dictionary.get_idx_for_item(STOP_TAG)], device=flair.device
		)
		stop = stop[None, :].repeat(tags.shape[0], 1)

		pad_start_tags = torch.cat([start, tags], 1).cuda()
		pad_stop_tags = torch.cat([tags, stop], 1).cuda()
		transition_mask=torch.ones(mask.shape[0],mask.shape[1]+1).type_as(mask)
		transition_mask[:,1:]=mask
		transition_mask2=torch.ones(mask.shape[0],mask.shape[1]+1).type_as(mask)
		transition_mask2[:,:-1]=mask
		transition_mask2[:,-1]=0
		try:
			pad_stop_tags = pad_stop_tags.cuda()*transition_mask2.long()+(1-transition_mask2.long())*self.tag_dictionary.get_idx_for_item(STOP_TAG)
		except:
			pdb.set_trace()
		
		my_emission=torch.gather(feats,2,tags.unsqueeze(-1))*mask.unsqueeze(-1)
		
		my_emission=my_emission.sum(-1).sum(-1)
		# (bat_size, seq_len + 1, target_size_to, target_size_from)
		bat_size=feats.shape[0]
		seq_len=pad_stop_tags.shape[1]
		ts_energy=self.transitions.unsqueeze(0).unsqueeze(0).expand(bat_size,seq_len,self.tagset_size,self.tagset_size)

		# extract the first dimension (2nd dimension here) of transition scores
		# (bat_size, seq_len + 1, target_size_to, target_size_from) -> (bat_size, seq_len + 1, 1, target_size_from)
		ts_energy=torch.gather(ts_energy,2,pad_stop_tags.unsqueeze(-1).unsqueeze(-1).expand(bat_size,seq_len,1,feats.shape[-1]))
		# (bat_size, seq_len + 1, 1, target_size_from) -> (bat_size, seq_len + 1, target_size_from)
		ts_energy=ts_energy.squeeze(2)
		# (bat_size, seq_len + 1, target_size_from) -> (bat_size, seq_len + 1)
		ts_energy=torch.gather(ts_energy,2,pad_start_tags.unsqueeze(-1)).squeeze(-1)
		
		ts_energy=ts_energy*transition_mask
		ts_energy=ts_energy.sum(1)
		score=ts_energy+my_emission
		# my_transition=
		return score

	def evaluate(
		self,
		data_loader: DataLoader,
		out_path: Path = None,
		embeddings_storage_mode: str = "cpu",
		prediction_mode = False,
		speed_test = False,
	) -> (Result, float):
		with torch.no_grad():
			eval_loss = 0

			batch_no: int = 0

			metric = Metric("Evaluation")

			lines: List[str] = []
			if out_path is not None:
				outfile = open(out_path, "w", encoding="utf-8")
			if speed_test:
				start_time = time.time()
			for batch in data_loader:
				batch_no += 1

				with torch.no_grad():
					features = self.forward(batch,prediction_mode=prediction_mode)
					if not speed_test:
						mask=self.mask
						loss = self._calculate_loss(features, batch, mask)
						tags, _ = self._obtain_labels(features, batch)
					else:
						self._obtain_labels_speed(features, batch)
				if not speed_test:
					eval_loss += loss

					for (sentence, sent_tags) in zip(batch, tags):
						for (token, tag) in zip(sentence.tokens, sent_tags):
							token: Token = token
							token.add_tag_label("predicted", tag)

							# append both to file for evaluation
							eval_line = "{} {} {} {}\n".format(
								token.text,
								token.get_tag(self.tag_type).value,
								tag.value,
								tag.score,
							)
							# lines.append(eval_line)
							if out_path is not None:
								outfile.write(eval_line)
						# lines.append("\n")
						if out_path is not None:
							outfile.write("\n")
					for sentence in batch:
						# make list of gold tags
						gold_tags = [
							(tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)
						]
						# make list of predicted tags
						predicted_tags = [
							(tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
						]

						# check for true positives, false positives and false negatives
						for tag, prediction in predicted_tags:
							if (tag, prediction) in gold_tags:
								metric.add_tp(tag)
							else:
								metric.add_fp(tag)

						for tag, gold in gold_tags:
							if (tag, gold) not in predicted_tags:
								metric.add_fn(tag)
							else:
								metric.add_tn(tag)

						# pdb.set_trace()
					if prediction_mode and batch_no % (len(data_loader)//10) == 0:
						log.info(f"{batch_no}/{len(data_loader)}")
				store_embeddings(batch, embeddings_storage_mode)
				if embeddings_storage_mode == "none":
					del batch.features
			if speed_test:
				end_time = time.time()
				print(data_loader.num_examples/(end_time-start_time))
			eval_loss /= batch_no
			if out_path is not None:
				outfile.close()
			# if out_path is not None:
			# 	with open(out_path, "w", encoding="utf-8") as outfile:
			# 		outfile.write("".join(lines))

			detailed_result = (
				f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
				f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
			)
			for class_name in metric.get_classes():
				detailed_result += (
					f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
					f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
					f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
					f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
					f"{metric.f_score(class_name):.4f}"
				)

			result = Result(
				main_score=metric.micro_avg_f_score(),
				log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
				log_header="PRECISION\tRECALL\tF1",
				detailed_results=detailed_result,
			)

			return result, eval_loss
