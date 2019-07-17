import os
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('seqs', metavar='SEQS_PATH', help='path to seqs')
parser.add_argument('labels', metavar='LABELS_PATH', help='path to labels')
parser.add_argument('output_dir', metavar='OUTPUT_DIR', help='output directory')
parser.add_argument('--name', type=str, default='', help='name of dataset. Default=None')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')


def load_data(path_seq, path_label):
	with open(path_seq, 'rb') as f:
		seqs = pickle.load(f)
	with open(path_label, 'rb') as f:
		labels = pickle.load(f)
	return seqs, labels


def partition(n, testRatio=0.2, validRatio=0.1, seed=0):
	np.random.seed(seed)
	ind = np.random.permutation(n)
	nTest = int(testRatio * n)
	nValid = int(validRatio * n)
	return ind[nTest + nValid:], ind[0:nTest], ind[nTest:nTest + nValid]  # Train, Test, Validation


def aggregate_seqs(seqs):
	row = []
	col = []
	data = []

	# To count the number? max value? of codes
	max_code = 0

	for i, patient in enumerate(seqs):
		for visit in patient:
			for code in visit:
				if code > max_code:
					max_code = code
				row.append(i)
				col.append(code)
				data.append(1)

	aggregated = csr_matrix((data, (row, col)), shape=(len(seqs), max_code + 1))

	return aggregated


if __name__ == '__main__':
	args = parser.parse_args()
	if args.name != '':
		args.name = args.name + '_'

	seqs, labels = load_data(path_seq=args.seqs, path_label=args.labels)

	if len(seqs) != len(labels):
		raise ValueError("Visit sequences and labels have different lengths")

	num_samples = len(labels)
	train_index, test_index, valid_index = partition(num_samples, testRatio=0.2, validRatio=0.1)

	train_seqs = [seqs[i] for i in train_index]
	test_seqs = [seqs[i] for i in test_index]
	valid_seqs = [seqs[i] for i in valid_index]

	train_labels = [labels[i] for i in train_index]
	test_labels = [labels[i] for i in test_index]
	valid_labels = [labels[i] for i in valid_index]

	aggregated = aggregate_seqs(seqs)
	train_aggr = aggregated[train_index]
	test_aggr = aggregated[test_index]
	valid_aggr = aggregated[valid_index]

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	with open(args.output_dir + args.name + 'all.data_csr', 'wb') as f:
		pickle.dump(aggregated, f, pickle.HIGHEST_PROTOCOL)
	with open(args.output_dir + args.name + 'all.labels', 'wb') as f:
		pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

	with open(args.output_dir + args.name + 'train.seqs', 'wb') as f:
		pickle.dump(train_seqs, f, pickle.HIGHEST_PROTOCOL)
	with open(args.output_dir + args.name + 'train.data_csr', 'wb') as f:
		pickle.dump(train_aggr, f, pickle.HIGHEST_PROTOCOL)
	with open(args.output_dir + args.name + 'train.labels', 'wb') as f:
		pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)

	with open(args.output_dir + args.name + 'valid.seqs', 'wb') as f:
		pickle.dump(valid_seqs, f, pickle.HIGHEST_PROTOCOL)
	with open(args.output_dir + args.name + 'valid.data_csr', 'wb') as f:
		pickle.dump(valid_aggr, f, pickle.HIGHEST_PROTOCOL)
	with open(args.output_dir + args.name + 'valid.labels', 'wb') as f:
		pickle.dump(valid_labels, f, pickle.HIGHEST_PROTOCOL)

	with open(args.output_dir + args.name + 'test.seqs', 'wb') as f:
		pickle.dump(test_seqs, f, pickle.HIGHEST_PROTOCOL)
	with open(args.output_dir + args.name + 'test.data_csr', 'wb') as f:
		pickle.dump(test_aggr, f, pickle.HIGHEST_PROTOCOL)
	with open(args.output_dir + args.name + 'test.labels', 'wb') as f:
		pickle.dump(test_labels, f, pickle.HIGHEST_PROTOCOL)

	''' Save Normalized Data Also '''
	scaler = MaxAbsScaler()
	train_aggr = scaler.fit_transform(train_aggr)
	with open(args.output_dir + args.name + "scaler.pkl", "wb") as f:
		pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)
	valid_aggr = scaler.transform(valid_aggr)
	test_aggr = scaler.transform(test_aggr)

	with open(args.output_dir + args.name + 'train_normalized.data_csr', 'wb') as f:
		pickle.dump(train_aggr, f, pickle.HIGHEST_PROTOCOL)
	with open(args.output_dir + args.name + 'valid_normalized.data_csr', 'wb') as f:
		pickle.dump(valid_aggr, f, pickle.HIGHEST_PROTOCOL)
	with open(args.output_dir + args.name + 'test_normalized.data_csr', 'wb') as f:
		pickle.dump(test_aggr, f, pickle.HIGHEST_PROTOCOL)
