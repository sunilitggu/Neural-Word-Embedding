from __future__ import division
from numpy import random, zeros, eye, array, ravel, dot, matlib, exp, max, sum, log, multiply, divide, add, subtract
from math import floor
import cPickle
from scipy.io import loadmat
#THIS IS A PYTHON PORT OF A MATLAB CODE PROVIDED AS PART OF AN ASSIGNMENT FROM CSC 321 - ANTHONY BONNER, UNIVERSITY OF TORONTO
#THIS CODE WAS PRODUCED IN AN EFFORT TO NOT RELY ON MATLAB
#THE ORIGINAL ASSIGNMENT CAN BE FOUND AT: http://www.cs.toronto.edu/~nitish/csc321/assignment1.html
#THIS MODEL COMPUTES A 'NEURAL' WORD EMBEDDING FOR EACH WORD IN THE VOCABULARY
#FOR DETAILS PLEASE VISIT ASSIGNMENT WEBSITE  

#LOAD DATA FROM ASSIGNMENT TO TEST CORRECTNESS (data.mat can be found in the download for the original assignment)
data = {}
loadmat('data.mat', data)
train_input = data['data']['trainData'][0][0][:3, :]-1 # train_input is a matrix of trigrams where each trigram = [index_of_word_1, index_of_word_2, index_of_word_3]
train_target = data['data']['trainData'][0][0][3, :]-1 # train_target a matrix of the indices of words that come after each trigram
valid_input = data['data']['validData'][0][0][:3, :]-1 
valid_target = data['data']['validData'][0][0][3, :]-1
vocab = data['data']['vocab'][0] #Word array
vocab = [str(x[0]) for x in vocab[0][0]]


#data = cPickle.load(open('data', 'r'))
#train_input = data[0] 
#train_target = data[1]
#valid_input = data[2]
#valid_target = data[3]
#vocab = data[4]

batchsize = 100
learning_rate = 0.1
momentum = 0.9
weight_cost = 0.0
epochs = 50
numhid1 = 8
numhid2 = 64
init_wt = 0.01
tiny = exp(-30)
K = len(vocab)
D = 3
show_training_CE_after = 100
show_validation_CE_after = 1000

numbatches = int(floor(len(train_input[0])/batchsize))
word_embedding_weights = random.normal(0.0, init_wt, size=(K, numhid1))
embed_to_hid_weights = random.normal(0.0, init_wt, size=(D*numhid1, numhid2))
hid_to_output_weights = random.normal(0.0, init_wt, (numhid2, K))
hid_bias = zeros([numhid2, 1])
output_bias = zeros([K, 1])
word_embedding_weights_delta = zeros([K, numhid1])
word_embedding_weights_grad = zeros([K, numhid1])
embed_to_hid_weights_delta = zeros([D * numhid1, numhid2])
hid_to_output_weights_delta = zeros([numhid2, K])
hid_bias_delta = zeros([numhid2, 1])
output_bias_delta = zeros([K, 1])
expansion_matrix = eye(K)
best_valid_CE = float("inf")
end_training = False
for e in range(epochs):
	this_chunk_CE = 0
	count = 0
	for m in range(numbatches):
		#FEED FORWARD
		input_batch = train_input[:, (m)*batchsize:(m+1)*batchsize]
		target_batch = train_target[(m)*batchsize:(m+1)*batchsize]
		embedding_layer_state = word_embedding_weights[input_batch.T, :]
		embedding_layer_state = array([ravel(x) for x in embedding_layer_state]).T
		inputs_to_hid = dot(embed_to_hid_weights.T, embedding_layer_state) + matlib.repmat(hid_bias, 1, batchsize)
		hidden_layer_state = divide(1, (1 + exp(multiply(-1.0, inputs_to_hid))))
		inputs_to_softmax = dot(hid_to_output_weights.T, hidden_layer_state) + matlib.repmat(output_bias, 1, batchsize)
		inputs_to_softmax = subtract(inputs_to_softmax, matlib.repmat(max(inputs_to_softmax), K, 1))
		output_layer_state = exp(inputs_to_softmax)
		output_layer_state = divide(output_layer_state, matlib.repmat(sum(output_layer_state, 0), K, 1))
		expanded_target_batch = expansion_matrix[:, target_batch]
		error_deriv = subtract(output_layer_state, expanded_target_batch)
		CE = divide(multiply(-1.0, sum(sum(multiply(expanded_target_batch, log(add(output_layer_state, tiny)))))), batchsize)
		count =  count + 1
		this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count;
		if (m % show_training_CE_after) == 0:
			print 'Batch: ', m, ' CE: ', this_chunk_CE
			count = 0
			this_chunk_CE = 0
		hid_to_output_weights_grad = dot(hidden_layer_state, error_deriv.T)
		output_bias_grad = array([sum(error_deriv, 1)]).T
		back_prop_deriv_1 = multiply(multiply(dot(hid_to_output_weights, error_deriv), hidden_layer_state), subtract(1, hidden_layer_state))
		embed_to_hid_weights_grad = dot(embedding_layer_state, back_prop_deriv_1.T)
		hid_bias_grad = array([sum(back_prop_deriv_1, 1)]).T
		back_prop_deriv_2 = dot(embed_to_hid_weights, back_prop_deriv_1)
		word_embedding_weights_grad[:] = 0
		for w in range(D):
			word_embedding_weights_grad = add(word_embedding_weights_grad, dot(expansion_matrix[:, input_batch[w, :]], (back_prop_deriv_2[w*numhid1 : (w+1)*numhid1, :]).T))
		word_embedding_weights_delta = add(multiply(momentum, word_embedding_weights_delta), divide(word_embedding_weights_grad, batchsize))
		word_embedding_weights = subtract(word_embedding_weights, multiply(learning_rate, word_embedding_weights_delta))
		embed_to_hid_weights_delta = add(multiply(momentum, embed_to_hid_weights_delta), divide(embed_to_hid_weights_grad, batchsize))
		embed_to_hid_weights = subtract(embed_to_hid_weights, multiply(learning_rate, embed_to_hid_weights_delta))
		hid_to_output_weights_delta = add(multiply(momentum, hid_to_output_weights_delta), divide(hid_to_output_weights_grad, batchsize))
		hid_to_output_weights = subtract(hid_to_output_weights, multiply(learning_rate, hid_to_output_weights_delta))
		hid_bias_delta = add(multiply(momentum, hid_bias_delta), divide(hid_bias_grad, batchsize))
		hid_bias = subtract(hid_bias, multiply(learning_rate, hid_bias_delta))
		output_bias_delta = add(multiply(momentum, output_bias_delta), divide(output_bias_grad, batchsize))
		output_bias = subtract(output_bias, multiply(learning_rate, output_bias_delta))
		if (m % show_validation_CE_after) == 0:
			print 'validating...'
			input_size = len(valid_input[0])
			n_b = int(floor(input_size/batchsize))
			e_m = eye(K)
			c = 0
			avg_CE = 0.0
			for m_m in range(n_b):
				i_b = valid_input[(m_m)*batchsize:(m_m+1)*batchsize]
				t_b = valid_target[(m_m)*batchsize:(m_m+1)*batchsize]
				e_l_s = word_embedding_weights[i_b.T, :]
				e_l_s = array([ravel(x) for x in embedding_layer_state])
				i_t_h = dot(embed_to_hid_weights.T, e_l_s) + matlib.repmat(hid_bias, 1, batchsize)
				h_l_s = divide(1, (1 + exp(multiply(-1.0, i_t_h))))
				i_t_s = dot(hid_to_output_weights.T, h_l_s) + matlib.repmat(output_bias, 1, batchsize)
				i_t_s = subtract(i_t_s, matlib.repmat(max(i_t_s), K, 1))
				o_l_s = exp(i_t_s)
				o_l_s = divide(o_l_s, matlib.repmat(sum(o_l_s, 0), K, 1))
				e_t_b = expansion_matrix[:, t_b]
				C_E = divide(multiply(-1.0, sum(sum(multiply(e_t_b, log(add(o_l_s, tiny)))))), batchsize)
				c = c + 1
				avg_CE = avg_CE + (C_E - avg_CE) / c
			print 'Validation CE: ', avg_CE
			if avg_CE > best_valid_CE:
				end_training = True
				break
			best_valid_CE = avg_CE
	if end_training:
		cPickle.dump([word_embedding_weights, vocab], open('weights_vocab', 'w'))
		break
	cPickle.dump([word_embedding_weights, vocab], open('weights_vocab', 'w'))
