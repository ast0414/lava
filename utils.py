import numpy as np
from scipy.sparse import coo_matrix

import torch
from torch.utils.data import Dataset


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


""" Custom Dataset """


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features, reverse):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
			reverse (bool): If true, reverse the order of sequence (for RETAIN)
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.seqs = []
		# self.labels = []

		for seq, label in zip(seqs, labels):

			if reverse:
				sequence = list(reversed(seq))
			else:
				sequence = seq

			row = []
			col = []
			val = []
			for i, visit in enumerate(sequence):
				for code in visit:
					if code < num_features:
						row.append(i)
						col.append(code)
						val.append(1.0)

			self.seqs.append(coo_matrix((np.array(val, dtype=np.float32), (np.array(row), np.array(col))), shape=(len(sequence), num_features)))
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.seqs[index], self.labels[index]


""" Custom collate_fn for DataLoader"""


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a SparseFloatTensor, and label is a LongTensor

	:returns
		seqs
		labels
		lengths
		indices
	"""
	batch_seq, batch_label = zip(*batch)

	num_features = batch_seq[0].shape[1]
	seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
	max_length = max(seq_lengths)

	sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
	sorted_padded_seqs = []
	sorted_labels = []

	for i in sorted_indices:
		length = batch_seq[i].shape[0]

		if length < max_length:
			padded = np.concatenate(
				(batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
		else:
			padded = batch_seq[i].toarray()

		sorted_padded_seqs.append(padded)
		sorted_labels.append(batch_label[i])

	seq_tensor = np.stack(sorted_padded_seqs, axis=0)
	label_tensor = torch.LongTensor(sorted_labels)

	return torch.from_numpy(seq_tensor), label_tensor, list(sorted_lengths), list(sorted_indices)


def batch_patient_tensor_to_list(batch_tensor, lengths, reverse):

	batch_size, max_len, num_features = batch_tensor.size()
	patients_list = []

	for i in range(batch_size):
		patient = []
		for j in range(lengths[i]):
			codes = torch.nonzero(batch_tensor[i][j])
			if codes.is_cuda:
				codes = codes.cpu()
			patient.append(sorted(codes.numpy().flatten().tolist()))

		if reverse:
			patients_list.append(list(reversed(patient)))
		else:
			patients_list.append(patient)

	return patients_list
