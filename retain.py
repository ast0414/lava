import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import AverageMeter

from tqdm import tqdm


""" RETAIN model class """


class RETAIN(nn.Module):
	def __init__(self, dim_input, dim_emb=128, dropout_emb=0.6, dim_alpha=128, dim_beta=128, dropout_context=0.6, dim_output=2, batch_first=True):
		super(RETAIN, self).__init__()
		self.batch_first = batch_first
		self.embedding = nn.Sequential(
			nn.Linear(dim_input, dim_emb, bias=False),
			nn.Dropout(p=dropout_emb)
		)
		init.xavier_normal(self.embedding[0].weight)

		self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)

		self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
		init.xavier_normal(self.alpha_fc.weight)
		self.alpha_fc.bias.data.zero_()

		self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)

		self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
		init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
		self.beta_fc.bias.data.zero_()

		self.output = nn.Sequential(
			nn.Dropout(p=dropout_context),
			nn.Linear(in_features=dim_emb, out_features=dim_output)
		)
		init.xavier_normal(self.output[1].weight)
		self.output[1].bias.data.zero_()

	def get_W(self):
		return self.output[1].weight

	def get_Wemb(self):
		return self.embedding[0].weight

	def forward(self, x, lengths):
		if self.batch_first:
			batch_size, max_len = x.size()[:2]
		else:
			max_len, batch_size = x.size()[:2]

		# emb -> batch_size X max_len X dim_emb
		emb = self.embedding(x)

		packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)

		g, _ = self.rnn_alpha(packed_input)

		# alpha_unpacked -> batch_size X max_len X dim_alpha
		alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)

		# mask -> batch_size X max_len X 1
		mask = Variable(torch.FloatTensor([[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2), requires_grad=False)
		if next(self.parameters()).is_cuda:  # returns a boolean
			mask = mask.cuda()

		# e => batch_size X max_len X 1
		e = self.alpha_fc(alpha_unpacked)

		def masked_softmax(batch_tensor, mask):
			exp = torch.exp(batch_tensor)
			masked_exp = exp * mask
			sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
			return masked_exp / sum_masked_exp

		# Alpha = batch_size X max_len X 1
		# alpha value for padded visits (zero) will be zero
		alpha = masked_softmax(e, mask)

		h, _ = self.rnn_beta(packed_input)

		# beta_unpacked -> batch_size X max_len X dim_beta
		beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)

		# Beta -> batch_size X max_len X dim_emb
		# beta for padded visits will be zero-vectors
		beta = F.tanh(self.beta_fc(beta_unpacked) * mask)

		# context -> batch_size X (1) X dim_emb (squeezed)
		# Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
		# Vectorized sum
		context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

		# without applying non-linearity
		logit = self.output(context)

		return logit, alpha, beta


""" Epoch function """


def retain_epoch(loader, model, criterion, optimizer=None, train=False):
	if train and not optimizer:
		raise AttributeError("Optimizer should be given for training")

	if train:
		model.train()
		mode = 'Train'
	else:
		model.eval()
		mode = 'Eval'

	losses = AverageMeter()
	labels = []
	outputs = []

	for bi, batch in enumerate(tqdm(loader, desc="{} batches".format(mode), leave=False)):
		inputs, targets, lengths, indices = batch

		input_var = torch.autograd.Variable(inputs)
		target_var = torch.autograd.Variable(targets)
		if next(model.parameters()).is_cuda:
			input_var = input_var.cuda()
			target_var = target_var.cuda()

		output, alpha, beta = model(input_var, lengths)
		loss = criterion(output, target_var)
		assert not np.isnan(loss.data[0]), 'Model diverged with loss = NaN'

		sorted_indices, original_indices = zip(*sorted(enumerate(indices), key=lambda x: x[1], reverse=False))
		idx = torch.LongTensor(sorted_indices)

		labels.append(targets.gather(0, idx))
		# since the outputs are logit, not probabilities
		outputs.append(F.softmax(output, dim=1).data.cpu().gather(0, torch.stack((idx, idx), dim=1)))

		# record loss
		losses.update(loss.data[0], inputs.size(0))

		# compute gradient and do update step
		if train:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg
