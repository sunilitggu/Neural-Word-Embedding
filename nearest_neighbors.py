from sklearn.neighbors import NearestNeighbors
import cPickle
import numpy as np
weights, vocab = cPickle.load(open('weights_vocab', 'r'))
nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(weights)

def get_n(s):
	index = vocab.index(s)
	distances, indices = nbrs.kneighbors(weights[index])
	for x in indices[0]:
		print vocab[x]


get_n('three')
