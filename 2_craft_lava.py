import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import VisitSequenceWithLabelDataset, visit_collate_fn, batch_patient_tensor_to_list


parser = argparse.ArgumentParser()
parser.add_argument('data_path', metavar='DATA_PATH', help='path to the clean dataset')
parser.add_argument('model_path', metavar='MODEL_PATH', help='path to the source model used to craft examples')

parser.add_argument('--num_features', type=int, default=4894, metavar='N', help='number of features (i.e., input dimension')
parser.add_argument('--max-perturb', type=int, default=20, help='maximum perturbations (default: 20)')
parser.add_argument('--no-stop', dest='early_stop', action='store_false', help='No early termination of crafting')
parser.add_argument('--cost', type=str, default='ATT', help='Cost method (default: attention')
parser.add_argument('--lamda', type=float, default=1.0, help='cost strength')

parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use (default: -1 = (multiprocessing.cpu_count()-1 or 1))')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
parser.add_argument('--save', default='./Crafted/', type=str, metavar='SAVE_PATH', help='path to save (default: ./Crafted/)')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='NOT use cuda')
parser.set_defaults(cuda=True, early_stop=True)


def compute_forward_derivatives(inputs, outputs):
	"""
	:param inputs: Tensor: batch_size X max_seq_length X num_features
	:param outputs: Tensor: batch_size X num_class
	:return: jacobian Tensor: num_class X batch_size X max_seq_length X num_features
	"""
	assert inputs.requires_grad

	num_classes = outputs.size()[1]

	jacobian = torch.zeros(num_classes, *inputs.size())
	grad_output = torch.zeros(*outputs.size())

	if inputs.is_cuda:
		grad_output = grad_output.cuda()
		jacobian = jacobian.cuda()

	for i in range(num_classes):
		zero_gradients(inputs)
		grad_output.zero_()
		grad_output[:, i] = 1
		outputs.backward(grad_output, retain_graph=True)
		jacobian[i] = inputs.grad.data

	return jacobian


def compute_saliency_score(input_seqs, outputs, target_class, normalize=True):

	# Jacobian: dim_output X 1 (batch_size) X seq_length X num_features
	jacobian = compute_forward_derivatives(input_seqs, outputs)

	all_sum = torch.sum(jacobian, 0).squeeze()

	target_saliency = jacobian[target_class].squeeze()
	other_saliency = all_sum - target_saliency

	score = torch.sign(other_saliency) * torch.min(torch.zeros_like(all_sum), target_saliency*other_saliency)

	squeezed_input = input_seqs.data.squeeze()

	one_clamped = torch.clamp(squeezed_input * score, max=0)
	zero_clamped = torch.clamp((torch.ones_like(squeezed_input) - squeezed_input) * score, min=0)

	saliency = one_clamped + zero_clamped

	if normalize:
		saliency = saliency / torch.max(torch.abs(saliency))

	return saliency


def compute_contribution_derivatives(inputs, outputs):
	"""
	:param inputs: Tensor: batch_size X max_seq_length X num_features
	:param outputs: Tensor: batch_size X max_seq_length X num_features
	:return: jacobian Tensor: num_class X batch_size X max_seq_length X num_features
	"""
	assert inputs.requires_grad

	grad_output = torch.ones(*outputs.size())
	if inputs.is_cuda:
		grad_output = grad_output.cuda()

	zero_gradients(inputs)
	outputs.backward(grad_output)

	return inputs.grad.data[0]


# TODO: Currently, assume one sample at each time
def lava(model, inputs, target_class, lengths, max_distortion=20, early_stop=False):
	# max_distortion = int : maximum code deviations (total number of increase/descrease)

	# Make a clone since we will alter the values
	# If we require grad, cuda first, then wrap it as Variable
	input_seqs = torch.autograd.Variable(inputs.clone(), requires_grad=True)
	max_iter = max_distortion  # Modifying 1 features at each iteration
	count = 0

	output_logit, alpha, beta = model(input_seqs, lengths)
	output_prob = F.softmax(output_logit, dim=1)
	_, source_class = torch.max(output_prob.data, 1)

	while count < max_iter:
		if early_stop and source_class.equal(target_class):
			break

		# S in Eq.(13)
		saliency_score = compute_saliency_score(input_seqs, output_prob, target_class, normalize=True)

		W_emb = model.get_Wemb()
		W = model.get_W()

		# The contribution/impact by each feature and visit, x_{i,k}, with Eq.(10) in the paper
		cont = alpha * torch.matmul(W, beta.transpose(0, 1).transpose(1, 2) * W_emb).transpose(0, 1)[target_class] * input_seqs

		# Eq.(11)
		d_cont = compute_contribution_derivatives(input_seqs, cont)

		# D in Eq.(13)
		cont_data = torch.abs(d_cont)
		detect_cost = cont_data / torch.max(cont_data)

		# Eq.(12)
		sdr = saliency_score / detect_cost

		if sdr.is_cuda:
			np_sdr = sdr.cpu().numpy()
		else:
			np_sdr = sdr.numpy()

		visit_index, feature_index = np.unravel_index(np.argmax(np.abs(np_sdr), axis=None), np_sdr.shape)
		opt_sdr = np_sdr[visit_index][feature_index]

		if opt_sdr > 0.0:
			input_seqs.data[0][visit_index][feature_index] += 1
		elif opt_sdr < 0.0:
			input_seqs.data[0][visit_index][feature_index] -= 1
		else:  # saliency == 0
			break

		output_logit, alpha, beta = model(input_seqs, lengths)
		output_prob = F.softmax(output_logit, dim=1)
		_, source_class = torch.max(output_prob.data, 1)

		count += 1

	return input_seqs


def craft_seq_adversarial_samples(data_loader, model, max_dist=10, early_stop=False):

	crafted_adv_seqs = []

	# switch to evaluation mode
	model.eval()

	for bi, batch in enumerate(tqdm(data_loader, desc="Crafting")):

		inputs, labels, lengths, indices = batch

		if args.cuda:
			inputs = inputs.cuda()
			labels = labels.cuda()
			# 'lengths' is a list, so no cuda

		# Assuming binary classification
		target_class = 1 - labels

		crafted = lava(model, inputs, target_class, lengths, max_distortion=max_dist, early_stop=early_stop)
		crafted_adv_seqs.extend(batch_patient_tensor_to_list(crafted.data, lengths, reverse=True))

	return crafted_adv_seqs


if __name__ == '__main__':
	args = parser.parse_args()

	if args.threads == -1:
		args.threads = torch.multiprocessing.cpu_count() - 1 or 1
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
	with open(args.data_path + 'test.seqs', 'rb') as f:
		test_seqs = pickle.load(f)
	with open(args.data_path + 'test.labels', 'rb') as f:
		test_labels = pickle.load(f)

	print("     ===> Construct a clean test set")
	# NOTE: reverse=True since we use RETAIN
	clean_test_set = VisitSequenceWithLabelDataset(test_seqs, test_labels, args.num_features, reverse=True)
	clean_loader = DataLoader(dataset=clean_test_set, batch_size=1, shuffle=False, collate_fn=visit_collate_fn, num_workers=args.threads)
	print('===> Dataset loaded!')

	# Create model
	print('===> Building a Model')
	source_model = torch.load(args.model_path)
	source_model = source_model.cpu()
	if args.cuda:
		source_model = source_model.cuda()

	print(source_model)
	print('===> Model built!')

	# Crafting Adversarial Examples
	print('===> Crafting Adversarial Examples with {} maximum perturbations'.format(args.max_perturb))
	seq_adv_samples_list = craft_seq_adversarial_samples(clean_loader, source_model, max_dist=args.max_perturb, early_stop=args.early_stop)

	if args.save:
		if not os.path.exists(args.save):
			os.makedirs(args.save)
		with open(os.path.join(args.save, 'adv_{}.seqs'.format(args.max_perturb)), 'wb') as f:
			pickle.dump(seq_adv_samples_list, f, pickle.HIGHEST_PROTOCOL)
	print('===> Adversarial Examples Crafted!')
