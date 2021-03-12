#!/usr/bin/env python
'''
Merge unireppytorch
'''

import _pickle as pk
import numpy as np
import sys
import os
import time
import torch
from torch import nn
import torch.utils.data as Data

TEST_MODE = False
NETNAME = os.path.splitext(os.path.basename(__file__))[0]


def aa_seq_to_int(s):
	aa_to_int = {
		'M':1,
		'R':2,
		'H':3,
		'K':4,
		'D':5,
		'E':6,
		'S':7,
		'T':8,
		'N':9,
		'Q':10,
		'C':11,
		'U':12,
		'G':13,
		'P':14,
		'A':15,
		'V':16,
		'I':17,
		'F':18,
		'Y':19,
		'W':20,
		'L':21,
		'O':22, #Pyrrolysine
		'X':23, # Unknown
		'Z':23, # Glutamic acid or GLutamine
		'B':23, # Asparagine or aspartic acid
		'J':23, # Leucine or isoleucine
		'start':24,
		'stop':25,
	}

	return [24] + [aa_to_int[a] for a in s]
#	return [24] + [aa_to_int[a] for a in s] + [25]


def is_valid_seq(seq, max_len=2000):
	l = len(seq)
	valid_aas = "MRHKDESTNQCUGPAVIFYWLO"
	if (l <= max_len) and set(seq) <= set(valid_aas):
		return True
	else:
		return False


class mLSTMCell(torch.nn.Module):
	def __init__(self,
			num_units,
			wx_init,
			wh_init,
			wmx_init,
			wmh_init,
			b_init,
			gx_init,
			gh_init,
			gmx_init,
			gmh_init,
			):
		super(mLSTMCell, self).__init__()
		self._num_units = num_units
		self.register_buffer("_wx", wx_init)
		self.register_buffer("_wh", wh_init)
		self.register_buffer("_wmx", wmx_init)
		self.register_buffer("_wmh", wmh_init)
		self.register_buffer("_b", b_init)
		self.register_buffer("_gx", gx_init)
		self.register_buffer("_gh", gh_init)
		self.register_buffer("_gmh", gmh_init)
		self.register_buffer("_gmx", gmx_init)

	def _apply(self, fn):
		super(mLSTMCell, self)._apply(fn)
		self._wx = fn(self._wx)
		self._wh = fn(self._wh)
		self._wmx = fn(self._wmx)
		self._wmh = fn(self._wmh)
		self._b = fn(self._b)
		self._gx = fn(self._gx)
		self._gh = fn(self._gh)
		self._gmx = fn(self._gmx)
		self._gmh = fn(self._gmh)
		return self

	def forward(self, inputs, state):
		c_prev, h_prev = state
		wx = torch.nn.functional.normalize(self._wx, dim=0, p=2) * self._gx
		wh = torch.nn.functional.normalize(self._wh, dim=0, p=2) * self._gh
		wmx = torch.nn.functional.normalize(self._wmx, dim=0, p=2) * self._gmx
		wmh = torch.nn.functional.normalize(self._wmh, dim=0, p=2) * self._gmh
		m = torch.matmul(inputs, wmx) * torch.matmul(h_prev, wmh)
		z = torch.matmul(inputs, wx) + torch.matmul(m, wh) + self._b
		i, f, o, u = torch.split(z, z.shape[1]//4, 1)
		i = torch.sigmoid(i)
		f = torch.sigmoid(f)
		o = torch.sigmoid(o)
		u = torch.tanh(u)
		c = f * c_prev + i * u
		h = o * torch.tanh(c)
		return h, (c, h)


class mLSTMCellStackNPY(torch.nn.Module):
	def __init__(self, num_units=64, num_layers=4, model_path="./"):
		super(mLSTMCellStackNPY, self).__init__()
		self._model_path=model_path
		self._num_units = num_units
		self._num_layers = num_layers
		bs = "rnn_mlstm_stack_mlstm_stack"
		join = lambda x: os.path.join(self._model_path, x)
		self._layers = torch.nn.ModuleList()
		for i in range(self._num_layers):
			self._layers.append(mLSTMCell(
			num_units=self._num_units,
			wx_init=torch.tensor(np.load(join(bs + "{0}_mlstm_stack{1}_wx:0.npy".format(i,i)))),
			wh_init=torch.tensor(np.load(join(bs + "{0}_mlstm_stack{1}_wh:0.npy".format(i,i)))),
			wmx_init=torch.tensor(np.load(join(bs + "{0}_mlstm_stack{1}_wmx:0.npy".format(i,i)))),
			wmh_init=torch.tensor(np.load(join(bs + "{0}_mlstm_stack{1}_wmh:0.npy".format(i,i)))),
			b_init=torch.tensor(np.load(join(bs + "{0}_mlstm_stack{1}_b:0.npy".format(i,i)))),
			gx_init=torch.tensor(np.load(join(bs + "{0}_mlstm_stack{1}_gx:0.npy".format(i,i)))),
			gh_init=torch.tensor(np.load(join(bs + "{0}_mlstm_stack{1}_gh:0.npy".format(i,i)))),
			gmx_init=torch.tensor(np.load(join(bs + "{0}_mlstm_stack{1}_gmx:0.npy".format(i,i)))),
			gmh_init=torch.tensor(np.load(join(bs + "{0}_mlstm_stack{1}_gmh:0.npy".format(i,i))))
			))

	def forward(self, inputs, state):
		c_prev, h_prev = state
		new_outputs = []
		new_cs = []
		new_hs = []
		for i, layer in enumerate(self._layers):
			if i == 0:
				h, (c,h_state) = layer(inputs, (c_prev[i],h_prev[i]))
			else:
				h, (c,h_state) = layer(new_outputs[-1], (c_prev[i],h_prev[i]))
			new_outputs.append(h)
			new_cs.append(c)
			new_hs.append(h_state)
		final_output = new_outputs[-1]
		return final_output, (tuple(new_cs), tuple(new_hs))


class babbler64(torch.nn.Module):
	def __init__(self, model_path="./64_weights/"):
		super(babbler64, self).__init__()
		self._rnn_size = 64
		self._vocab_size = 26
		self._embed_dim = 10
		self._num_layers = 4
		self._model_path = model_path
		self.rnn = mLSTMCellStackNPY(num_units=self._rnn_size, num_layers=self._num_layers, model_path=model_path)
#		self.state_zeros = torch.zeros([1, self._rnn_size], dtype=torch.float32)
		self.register_buffer("state_zeros", torch.zeros([1, self._rnn_size], dtype=torch.float32))

		self.embed_matrix = torch.tensor(np.load(os.path.join(self._model_path, "embed_matrix:0.npy")), requires_grad=False)
		self.embedlayer = torch.nn.Embedding(num_embeddings=self._vocab_size, embedding_dim=self._embed_dim, _weight=self.embed_matrix)

	def _apply(self, fn):
		super(babbler64, self)._apply(fn)
		self.state_zeros = fn(self.state_zeros)
		return self

	def forward(self, int_seq):
		state = (tuple(self.state_zeros.expand(int_seq.shape[0], -1) for _ in range(self._num_layers)), tuple(self.state_zeros.expand(int_seq.shape[0], -1) for _ in range(self._num_layers)))
		inp = self.embedlayer(int_seq).permute(1,0,2)
		hs_ = []
		for i in inp:
			output, state = self.rnn(i, state)
			hs_.append(output)
		hs = torch.stack(hs_, dim=1)
		return hs


class ResBlockV2(nn.Module):
	def __init__(self, n_in, n_out, half_win_size):
		super(ResBlockV2, self).__init__()
		self.n_in = n_in
		self.n_out = n_out
		self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=2*half_win_size+1, stride=1, padding=half_win_size, bias=True)
		self.batchnorm1 = nn.BatchNorm2d(n_out, affine=True)
		self.act1 = nn.ReLU()
		self.conv2 = nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=2*half_win_size+1, stride=1, padding=half_win_size, bias=True)
		self.batchnorm2 = nn.BatchNorm2d(n_out, affine=True)
		self.conv3 = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
		self.act2 = nn.ReLU()

	def forward(self, inp, mask):
		out = self.conv1(inp)
		out = self.batchnorm1(out)
		out = self.act1(out)
		out = out*mask
		out = self.conv2(out)
		out = self.batchnorm2(out)
		out = out*mask
		if(self.n_in==self.n_out):
			out = inp+out
		else:
			out = self.conv3(inp)+out
		out = self.act2(out)
		return out


class ResNet(nn.Module):
	def __init__(self, n_in, half_win_size, n_hiddens, n_repeats):
		super(ResNet,self).__init__()
		self.n_in = n_in
		self.ResBlock = nn.ModuleList()
		self.n_hiddens = n_hiddens
		self.n_repeats = n_repeats

		for i in range(len(n_hiddens)):
			for j in range(n_repeats[i]):
				n_out = n_hiddens[i]
				if i>=0 and j>0:
					n_in = n_hiddens[i]
				if i>0 and j==0:
					n_in = n_hiddens[i-1]
				self.ResBlock.append(ResBlockV2(n_in, n_out, half_win_size))

	def forward(self, inp, mask):
		num = 0
		for i in range(len(self.n_hiddens)):
			for j in range(self.n_repeats[i]):
				n_out = n_hiddens[i]
				if i>0 and j>0:
					self.n_in = self.n_hiddens[i]
				if i>0 and j==0:
					self.n_in = self.n_hiddens[i-1]
				inp = self.ResBlock[num](inp, mask)
				num = num+1
		return inp


class Linear(nn.Module):
	def __init__(self, features, n_middle, n_final):
		super(Linear, self).__init__()
		self.n_middle = n_middle
		self.n_final = n_final
		self.features = features
		self.midLinear = nn.Linear(self.features, self.n_middle)
		self.finalLinear = nn.Linear(self.n_middle, self.n_final)

	def forward(self, inp):
		out = inp.permute(0, 2, 3, 1)
		out = out.reshape(-1, self.features)
		out = self.midLinear(out)
		out = self.finalLinear(out)
		out = out.reshape(-1, self.n_final)
		return out


class Loss(nn.Module):
	def __init__(self):
		super(Loss, self).__init__()
		self.softmax = nn.Softmax(dim=1)
		self.loss_weight = torch.tensor([0.05, 0.95], requires_grad=False)
		self.loss = nn.CrossEntropyLoss(ignore_index=-1, weight=self.loss_weight)

	def _apply(self, fn):
		super(Loss, self)._apply(fn)
		self.loss_weight = fn(self.loss_weight)
		return self

	def forward(self, inp, target):
		out = self.softmax(inp)
		out = self.loss(out, target)
		return out


class data_preprocess(nn.Module):
	def __init__(self, unirep_size=64, maxseqlen=300):
		super(data_preprocess, self).__init__()
		self.requires_grad = False
		if (unirep_size == 64):
			model_path='./64_weights'
			self.urp = babbler64(model_path=model_path)
		elif (unirep_size == 256):
			model_path='./256_weights'
			self.urp = babbler256(model_path=model_path)
		elif (unirep_size == 1900):
			model_path='./1900_weights'
			self.urp = babbler1900(model_path=model_path)
		else:
			print("unirep size should be one of 64/256/1900")
			sys.exit(1)
		self.maxseqlen = maxseqlen 
		x = torch.meshgrid(torch.arange(maxseqlen), torch.arange(maxseqlen))
		self.y0 = x[0]
		self.y1 = x[1]
		self.y2 = (x[0] + x[1])//2

	def forward(self, seqs):
		if (type(seqs) is torch.Tensor):
			masks_1d = (seqs!=0)[:,1:].to(torch.float) #n*L
			reps_1d = self.urp(seqs)[:,:-1,:] * masks_1d.unsqueeze(2) #n*L*f
#			reps_1d = self.urp(seqs)[:,1:,:] * masks_1d.unsqueeze(2) #n*L*f

			masks_2d_x = masks_1d.unsqueeze(2).expand(-1,-1,self.maxseqlen) #n*L*L
			masks_2d_y = masks_1d.unsqueeze(1).expand(-1,self.maxseqlen,-1) #n*L*L
			masks_2d = (masks_2d_x * masks_2d_y).unsqueeze(1) #n*1*L*L

			reps_1d_t = reps_1d.permute((1, 0, 2)) #L*n*f
			reps_2d_0 = reps_1d_t[self.y0] #L*L*n*f
			reps_2d_1 = reps_1d_t[self.y1]
			reps_2d_2 = reps_1d_t[self.y2]
			reps_2d = torch.cat([reps_2d_0, reps_2d_1, reps_2d_2], dim=3) #L*L*n*3f
			reps_2d = reps_2d.permute(2, 3, 0, 1) * masks_2d #n*3f*L*L
		elif (type(seqs) is tuple):
			reps_2d, masks_2d = seqs
		else:
			sys.exit(1)

		return reps_2d, masks_2d


class Net(nn.Module):
	def __init__(self, n_in, half_win_size, n_hiddens, n_repeats, features, n_middle, n_final, unirep_size, maxseqlen):
		super(Net,self).__init__()
		self.ResNet=ResNet(n_in, half_win_size, n_hiddens, n_repeats)
		self.Linear=Linear(features, n_middle, n_final)
		self.dpp=data_preprocess(unirep_size, maxseqlen).requires_grad_(False)
		for param in self.ResNet.parameters():
			nn.init.normal_(param, mean=0, std=0.01)
		for param in self.Linear.parameters():
			nn.init.normal_(param, mean=0, std=0.01)

	def forward(self, seqs):
		out, mask = self.dpp(seqs)
		out = self.ResNet(out, mask)
		out = self.Linear(out)
		return out


def train(net, loss, train_iter, test_iter, optimizer, device, num_epochs):
	print("training on", device)
	net.to(device)
	loss.to(device)
	optimizer.zero_grad()
	for epoch in range(num_epochs):
		train_l_sum, contact_count, train_prec_sum, train_P_sum, train_acc_sum, n, batch_count, start_time = 0.0, 0.0, 0.0, 0.00000001, 0.0, 0, 0, time.time()
		for S, y in train_iter:
			chk_time = time.time()
			S = S.to(device)
			y = y.reshape(-1).to(device)
			y_real = (y!=-1)
			contact_count += (y==1).sum().item()#TP+FN
			y_hat = net(S)
			if (TEST_MODE): torch.cuda.synchronize(); print('Forward time: %.1f sec' % (time.time() - chk_time))
			l = loss(y_hat, y)
			l.backward()
			if (TEST_MODE): torch.cuda.synchronize(); print('Backward time: %.1f sec' % (time.time() - chk_time))
			optimizer.step()
			optimizer.zero_grad()
			if (TEST_MODE): torch.cuda.synchronize(); print('Step time: %.1f sec' % (time.time() - chk_time))
			train_l_sum += l.item()
			batch_count += 1
			y_pred = y_hat.argmax(dim=1)
			train_prec_sum += ((y_pred==1)*(y==1)).sum().item()#TP
			train_P_sum += ((y_pred==1)*y_real).sum().item()#TP+FP
			train_acc_sum += ((y_pred==y)*y_real).sum().item()#TP+TN
			n += y_real.sum().item()#TP+TN+FP+FN
			print('loss %.4f, contact ratio %.3f, train acc %.3f, train prec %.3f, train recall %.3f, time %.1f sec' % (train_l_sum / batch_count, contact_count / n, train_acc_sum / n, train_prec_sum / train_P_sum, train_prec_sum / contact_count, time.time() - chk_time))
		test_contact_ratio, test_acc, test_prec, test_recall = evaluate_accuracy(net, test_iter, device)
		print('epoch %d, loss %.4f, time %.1f sec:\ntrain contact ratio %.3f, train acc %.3f, train prec %.3f, train recall %.3f\ntest  contact ratio %.3f, test  acc %.3f, test  prec %.3f, test  recall %.3f' % (epoch + 1, train_l_sum / batch_count, time.time() - start_time, contact_count / n, train_acc_sum / n, train_prec_sum / train_P_sum, train_prec_sum / contact_count, test_contact_ratio, test_acc, test_prec, test_recall))
		if ((epoch + 1) % 1 == 0):
			torch.save(net, NETNAME+'.'+str(epoch+1)+'.model')
	return 0


def evaluate_accuracy(net, data_iter, device):
	contact_count, prec_sum, P_sum, acc_sum, n, start_time = 0.0, 0.0, 0.00000001, 0.0, 0, time.time()
	with torch.no_grad():
		for S, y in data_iter:
			net.eval()
			S = S.to(device)
			y = y.reshape(-1).to(device)
			y_real = (y!=-1)
			contact_count += (y==1).sum().item()#TP+FN
			y_pred = net(S).argmax(dim=1)
			prec_sum += ((y_pred==1)*(y==1)).sum().item()#TP
			P_sum += ((y_pred==1)*y_real).sum().item()#TP+FP
			acc_sum += ((y_pred==y)*y_real).sum().item()#TP+TN
			n += y_real.sum().item()#TP+TN+FP+FN
			net.train()
	if (TEST_MODE): print('Test time: %.1f sec' % (time.time() - start_time))
	return contact_count / n, acc_sum / n, prec_sum / P_sum, prec_sum / contact_count


def fileloader(dataset, type='pklfile', maxseqlen=300, batch_size=4):
	if (type=='pklfile'):
		pkl_input = open(dataset, 'rb')
		raw = pk.load(pkl_input, encoding='bytes')
		pkl_input.close()
		seqs_ = []
		cnts_ = []
		for t in raw:
			s=t[b'sequence'].decode()
			if(is_valid_seq(s, maxseqlen)):
				intseq = torch.tensor(aa_seq_to_int(s.strip()))
				seqs_.append(nn.functional.pad(intseq, (0, maxseqlen - intseq.shape[0] + 1), value=0))
				conmat = torch.tensor(t[b'contactMatrix'], dtype=torch.long)
				cnts_.append(nn.functional.pad(conmat, (0, maxseqlen - conmat.shape[1], 0, maxseqlen - conmat.shape[0]), value=-1))
		seqs = torch.stack(seqs_, dim=0)
		cnts = torch.stack(cnts_, dim=0)
		train_dataset = Data.TensorDataset(seqs, cnts)
		data_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	elif (type=='npzfile'):
		raw = np.load(data_file)
		total_input_2d, total_mask, total_target = raw.values()
		total_input_2d = torch.tensor(total_input_2d.astype(np.float32))
		total_mask = torch.tensor(np.expand_dims(total_mask, 1).astype(np.float32))
		total_target = torch.tensor(total_target).long()
		train_dataset = Data.TensorDataset((total_input_2d, total_mask), total_target)
		data_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	elif (type=='memory'):
		seqs, masks, cnts = dataset
		train_dataset = Data.TensorDataset(seqs, masks, cnts)
		data_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	else:
		sys.exit(1)

	return data_iter


if __name__=='__main__':
	if (TEST_MODE): start_time = time.time()

#	np.set_printoptions(threshold=sys.maxsize)
#	torch.set_printoptions(threshold=sys.maxsize)

#	device = torch.device("cpu")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_pkl = 'pdb25-6767-train.release.contactFeatures.pkl'
#	train_pkl = 'pdb25-test-500.release.contactFeatures.pkl'
	test_pkl = 'pdb25-6767-valid.release.contactFeatures.pkl'
#	test_pkl = 'pdb25-test-500.release.contactFeatures.pkl'

	inp_feature = 64
	maxseqlen = 300
	n_in = 3 * inp_feature
	half_win_size = 2
	n_hiddens = [50,55,60,65,70,75]
	n_repeats = [4,4,4,4,4,4]
	features = 75
	n_middle = 80
	n_final = 2
	lr = 0.001
	num_epochs = 20000
	train_batch_size = 8
	test_batch_size = 8

#	train_iter = fileloader('train366.out', 'npzfile', maxseqlen, train_batch_size)
#	test_iter = fileloader('test20.out', 'npzfile', maxseqlen, test_batch_size)
	train_iter = fileloader(train_pkl, 'pklfile', maxseqlen, train_batch_size)
	if (TEST_MODE): chk_time = time.time(); print('Walltime: %.1f seconds' % (chk_time - start_time))
	test_iter = fileloader(test_pkl, 'pklfile', maxseqlen, test_batch_size)
	if (TEST_MODE): chk_time = time.time(); print('Walltime: %.1f seconds' % (chk_time - start_time))

	net = Net(n_in, half_win_size, n_hiddens, n_repeats, features, n_middle, n_final, inp_feature, maxseqlen)
	net = nn.DataParallel(net)
#	for param in net.parameters():
#		nn.init.normal_(param, mean=0, std=0.01)

#	for mod in range(50,3700,50):
#		net = torch.load('nnet-2/nnet-'+str(mod)+'.model')
#		train_contact_ratio, train_acc, train_prec, train_recall = evaluate_accuracy(train_iter, net, device)
#		test_contact_ratio, test_acc, test_prec, test_recall = evaluate_accuracy(test_iter, net, device)
#		print('epoch %d:\ntrain contact ratio %.3f, train acc %.3f, train prec %.3f, train recall %.3f\ntest  contact ratio %.3f, test  acc %.3f, test  prec %.3f, test  recall %.3f' % (mod, train_contact_ratio, train_acc, train_prec, train_recall, test_contact_ratio, test_acc, test_prec, test_recall))

#	net = torch.load('nnet-2/nnet-3500.model')

	loss = Loss()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)

	if (TEST_MODE): chk_time = time.time(); print('Walltime: %.1f seconds' % (chk_time - start_time))
	train(net, loss, train_iter, test_iter, optimizer, device, num_epochs)

	torch.save(net, 'nnet-final.model')

	if (TEST_MODE): chk_time = time.time(); print('Walltime: %.1f seconds' % (chk_time - start_time))

