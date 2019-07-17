""" Matplotlib backend configuration """
import matplotlib
matplotlib.use('agg')  # generate postscript output by default

""" Imports """
import sys
import os
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import VisitSequenceWithLabelDataset, visit_collate_fn
from retain import RETAIN, retain_epoch


""" Arguments """
parser = argparse.ArgumentParser()

parser.add_argument('seqs_path', metavar='SEQS_PATH', help="path to the seq data")
parser.add_argument('labels_path', metavar='LABEL_PATH', help='path to true labels')
parser.add_argument('model_path', metavar='MODEL_PATH', help='path to the model to be evaluated')

parser.add_argument('--num_features', type=int, default=4894, metavar='N', help='number of features (i.e., input dimension')
parser.add_argument('--dim-emb', default=128, type=int, help='embedding dimension (default: 128)')
parser.add_argument('--drop-emb', default=0.5, type=float, help='embedding layer dropout rate (default: 0.5)')
parser.add_argument('--dim-alpha', default=128, type=int, help='RNN-Alpha hidden size (default: 128)')
parser.add_argument('--dim-beta', default=128, type=int, help='RNN-Beta hidden size (default: 128)')
parser.add_argument('--drop-context', default=0.5, type=float, help='context layer dropout rate (default: 0.5)')

parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='learning rate (default: 1e-2)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='mini-batch size for train (default: 256)')
parser.add_argument('-eb', '--eval-batch-size', type=int, default=256, help='mini-batch size for eval (default: 256)')

parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='NOT use cuda')
parser.add_argument('--no-plot', dest='plot', action='store_false', help='no plot')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use (default: 16)')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. (default: 1)')

parser.add_argument('--save', default='', type=str, metavar='SAVE_PATH', help='path to save results (default: SEQS_PATH)')
parser.add_argument('--resume', default='', type=str, metavar='LOAD_PATH', help='path to latest checkpoint (default: None)')

parser.set_defaults(cuda=True, plot=True)


""" Main function """


def main(argv):
	global args
	args = parser.parse_args(argv)
	if args.threads == -1:
		args.threads = torch.multiprocessing.cpu_count() - 1 or 1
	if args.save == '':
		args.save = os.path.dirname(args.seqs_path)
	print('===> Configuration')
	print(args)

	cuda = args.cuda
	if cuda:
		if torch.cuda.is_available():
			print('===> {} GPUs are available'.format(torch.cuda.device_count()))
		else:
			raise Exception("No GPU found, please run with --no-cuda")

	# Fix the random seed for reproducibility
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if cuda:
		torch.cuda.manual_seed(args.seed)

	# Data loading
	print('===> Loading test dataset')
	with open(args.seqs_path, 'rb') as f:
		test_seqs = pickle.load(f)
	with open(args.labels_path, 'rb') as f:
		test_labels = pickle.load(f)
	print("     ===> Num features: {}".format(args.num_features))

	test_set = VisitSequenceWithLabelDataset(test_seqs, test_labels, args.num_features, reverse=True)
	test_loader = DataLoader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=visit_collate_fn, num_workers=args.threads)
	print('===> Dataset loaded!')

	# Load model
	print('===> Loading a Model')
	model = torch.load(args.model_path)
	model = model.cpu()
	if cuda:
		model = model.cuda()
	print(model)
	print('===> Model built!')

	# No loss weight for test
	criterion = nn.CrossEntropyLoss()
	if args.cuda:
		criterion = criterion.cuda()

	if not os.path.exists(args.save):
		os.makedirs(args.save)

	# evaluate on the test set
	test_y_true, test_y_pred, test_loss = retain_epoch(test_loader, model, criterion=criterion)

	if args.cuda:
		test_y_true = test_y_true.cpu()
		test_y_pred = test_y_pred.cpu()

	test_auc = roc_auc_score(test_y_true.numpy(), test_y_pred.numpy()[:, 1], average="weighted")
	test_aupr = average_precision_score(test_y_true.numpy(), test_y_pred.numpy()[:, 1], average="weighted")

	with open(os.path.join(args.save, 'test_result.txt'), 'w') as f:
		f.write('Test Loss: {}\n'.format(test_loss))
		f.write('Test AUROC: {}\n'.format(test_auc))
		f.write('Test AUPR: {}\n'.format(test_aupr))

	print("Done!")
	print('Test Loss: {}\n'.format(test_loss))
	print('Test AUROC: {}\n'.format(test_auc))
	print('Test AUPR: {}\n'.format(test_aupr))


if __name__ == "__main__":
	main(sys.argv[1:])

