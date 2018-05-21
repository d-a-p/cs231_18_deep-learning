from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


# (afi_cache, bn_cache, reli_cache) => cache
def bn_forward(x, wi, bi, gamma, beta, bn_param):

	afi, afi_cache = affine_forward(x, wi, bi)
	reli, bn_cache = batchnorm_forward(afi, gamma, beta, bn_param)
	out, relu_cache = relu_forward(reli)
	cache = (afi_cache, bn_cache, relu_cache)
	return out, cache

def bn_backward(dout, cache):
	af_cache, bn_cache, relu_cache = cache

	dbn = relu_backward(dout, relu_cache)
	daf, dgamma, dbeta = batchnorm_backward_alt(dbn, bn_cache)
	dx, dw, db = affine_backward(daf, af_cache)
	return dx, dw, db, dgamma, dbeta

# (afi_cache, bn_cache, reli_cache) => cache
def ln_forward(x, wi, bi, gamma, beta, ln_param):

	afi, afi_cache = affine_forward(x, wi, bi)
	reli, ln_cache = layernorm_forward(afi, gamma, beta, ln_param)
	out, relu_cache = relu_forward(reli)
	cache = (afi_cache, ln_cache, relu_cache)
	return out, cache

def ln_backward(dout, cache):
	af_cache, ln_cache, relu_cache = cache

	dln = relu_backward(dout, relu_cache)
	daf, dgamma, dbeta = layernorm_backward(dln, ln_cache)
	dx, dw, db = affine_backward(daf, af_cache)
	return dx, dw, db, dgamma, dbeta

def ln_bn_forward(x, wi, bi, gamma, beta, n_param, normalization):
	xi, cache = None, None
	if normalization == 'batchnorm':
		xi, cache = bn_forward(x, wi, bi, gamma, beta, n_param)
	elif normalization == 'layernorm':
		xi, cache = ln_forward(x, wi, bi, gamma, beta, n_param)

	return xi, cache


def ln_bn_backward(dout, cache, normalization):
	dx, dwi, dbi, dgamma, dbeta = None, None, None, None, None
	if normalization == 'batchnorm':
		dx, dwi, dbi, dgamma, dbeta = bn_backward(dout, cache)
	elif normalization == 'layernorm':
		dx, dwi, dbi, dgamma, dbeta = ln_backward(dout, cache)

	return dx, dwi, dbi, dgamma, dbeta


class TwoLayerNet(object):
	"""
	A two-layer fully-connected neural network with ReLU nonlinearity and
	softmax loss that uses a modular layer design. We assume an input dimension
	of D, a hidden dimension of H, and perform classification over C classes.

	The architecure should be affine - relu - affine - softmax.

	Note that this class does not implement gradient descent; instead, it
	will interact with a separate Solver object that is responsible for running
	optimization.

	The learnable parameters of the model are stored in the dictionary
	self.params that maps parameter names to numpy arrays.
	"""

	def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
				 weight_scale=1e-3, reg=0.0):
		"""
		Initialize a new network.

		Inputs:
		- input_dim: An integer giving the size of the input
		- hidden_dim: An integer giving the size of the hidden layer
		- num_classes: An integer giving the number of classes to classify
		- weight_scale: Scalar giving the standard deviation for random
		  initialization of the weights.
		- reg: Scalar giving L2 regularization strength.
		"""
		self.params = {}
		self.reg = reg
		params = {}
		params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
		params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
		params['b1'] = np.zeros(hidden_dim)
		params['b2'] = np.zeros(num_classes)

		self.params = params

		############################################################################
		# Initialize the weights and biases of the two-layer net. Weights          #
		# should be initialized from a Gaussian centered at 0.0 with               #
		# standard deviation equal to weight_scale, and biases should be           #
		# initialized to zero. All weights and biases should be stored in the      #
		# dictionary self.params, with first layer weights                         #
		# and biases using the keys 'W1' and 'b1' and second layer                 #
		# weights and biases using the keys 'W2' and 'b2'.                         #
		############################################################################


	def loss(self, X, y=None):
		"""
		Compute loss and gradient for a minibatch of data.

		Inputs:
		- X: Array of input data of shape (N, d_1, ..., d_k)
		- y: Array of labels, of shape (N,). y[i] gives the label for X[i].

		Returns:
		If y is None, then run a test-time forward pass of the model and return:
		- scores: Array of shape (N, C) giving classification scores, where
		  scores[i, c] is the classification score for X[i] and class c.

		If y is not None, then run a training-time forward and backward pass and
		return a tuple of:
		- loss: Scalar value giving the loss
		- grads: Dictionary with the same keys as self.params, mapping parameter
		  names to gradients of the loss with respect to those parameters.
		"""
		scores = None
		w1 = self.params['W1']
		w2 = self.params['W2']
		b1 = self.params['b1']
		b2 = self.params['b2']
		reg = self.reg
		############################################################################
		# Implement the forward pass for the two-layer net, computing the          #
		# class scores for X and storing them in the scores variable.              #
		############################################################################
		af1, af1_cache = affine_forward(X, w1, b1)
		rel, rel_cache = relu_forward(af1)
		af2, af2_cache = affine_forward(rel, w2, b2)
		scores = af2

		# If y is None then we are in test mode so just return scores
		if y is None:
			return scores

		loss, grads = 0, {}

		num_loss, dscores = softmax_loss(scores, y)
		reg_loss = 0.5*reg*np.sum(w1*w1) + 0.5*reg*np.sum(w2*w2)
		loss = num_loss + reg_loss

		d_af2 = dscores
		d_rel, dw2, db2 = affine_backward(d_af2, af2_cache)
		d_af1 = relu_backward(d_rel, rel_cache)
		dx, dw1, db1 = affine_backward(d_af1, af1_cache)

		############################################################################
		# Implement the backward pass for the two-layer net. Store the loss        #
		# in the loss variable and gradients in the grads dictionary. Compute data #
		# loss using softmax, and make sure that grads[k] holds the gradients for  #
		# self.params[k]. Don't forget to add L2 regularization!                   #
		#                                                                          #
		# NOTE: To ensure that your implementation matches ours and you pass the   #
		# automated tests, make sure that your L2 regularization includes a factor #
		# of 0.5 to simplify the expression for the gradient.                      #
		############################################################################
		grads['W1'] = dw1 + (reg * w1)
		grads['W2'] = dw2 + (reg * w2)
		grads['b1'] = db1
		grads['b2'] = db2

		return loss, grads


class FullyConnectedNet(object):
	"""
	A fully-connected neural network with an arbitrary number of hidden layers,
	ReLU nonlinearities, and a softmax loss function. This will also implement
	dropout and batch/layer normalization as options. For a network with L layers,
	the architecture will be

	{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

	where batch/layer normalization and dropout are optional, and the {...} block is
	repeated L - 1 times.

	Similar to the TwoLayerNet above, learnable parameters are stored in the
	self.params dictionary and will be learned using the Solver class.
	"""

	def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
				 dropout=1, normalization=None, reg=0.0,
				 weight_scale=1e-2, dtype=np.float32, seed=None):
		"""
		Initialize a new FullyConnectedNet.

		Inputs:
		- hidden_dims: A list of integers giving the size of each hidden layer.
		- input_dim: An integer giving the size of the input.
		- num_classes: An integer giving the number of classes to classify.
		- dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
		  the network should not use dropout at all.
		- normalization: What type of normalization the network should use. Valid values
		  are "batchnorm", "layernorm", or None for no normalization (the default).
		- reg: Scalar giving L2 regularization strength.
		- weight_scale: Scalar giving the standard deviation for random
		  initialization of the weights.
		- dtype: A numpy datatype object; all computations will be performed using
		  this datatype. float32 is faster but less accurate, so you should use
		  float64 for numeric gradient checking.
		- seed: If not None, then pass this random seed to the dropout layers. This
		  will make the dropout layers deteriminstic so we can gradient check the
		  model.
		"""
		self.normalization = normalization
		self.use_dropout = dropout != 1
		self.reg = reg
		self.num_layers = 1 + len(hidden_dims)
		self.dtype = dtype
		self.params = {}
		params = {}

		for i in range(self.num_layers):
			ith_str = str(i+1) #store as 1 bigger as string in keys of params dict
			w_xdim = input_dim
			w_ydim = num_classes
			if i>0:
				w_xdim = hidden_dims[i-1]
			if i < len(hidden_dims):
				w_ydim = hidden_dims[i]

			if (self.normalization == 'batchnorm' or self.normalization == 'layernorm') and i > 0:
				params['gamma'+str(i)] = np.ones(hidden_dims[i-1])
				params['beta'+str(i)] = np.zeros(hidden_dims[i-1])

			params['W' + ith_str] = np.random.normal(0, weight_scale, (w_xdim, w_ydim))
			params['b' + ith_str] = np.zeros(w_ydim)

		self.params = params

		############################################################################
		# Initialize the parameters of the network, storing all values in          #
		# the self.params dictionary. Store weights and biases for the first layer #
		# in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
		# initialized from a normal distribution centered at 0 with standard       #
		# deviation equal to weight_scale. Biases should be initialized to zero.   #
		#                                                                          #
		# When using batch normalization, store scale and shift parameters for the #
		# first layer in gamma1 and beta1; for the second layer use gamma2 and     #
		# beta2, etc. Scale parameters should be initialized to ones and shift     #
		# parameters should be initialized to zeros.                               #
		############################################################################

		# When using dropout we need to pass a dropout_param dictionary to each
		# dropout layer so that the layer knows the dropout probability and the mode
		# (train / test). You can pass the same dropout_param to each dropout layer.
		self.dropout_param = {}
		if self.use_dropout:
			self.dropout_param = {'mode': 'train', 'p': dropout}
			if seed is not None:
				self.dropout_param['seed'] = seed

		# With batch normalization we need to keep track of running means and
		# variances, so we need to pass a special bn_param object to each batch
		# normalization layer. You should pass self.bn_params[0] to the forward pass
		# of the first batch normalization layer, self.bn_params[1] to the forward
		# pass of the second batch normalization layer, etc.
		self.bn_params = []
		if self.normalization=='batchnorm':
			self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
		if self.normalization=='layernorm':
			self.bn_params = [{} for i in range(self.num_layers - 1)]

		# Cast all parameters to the correct datatype
		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)


	def loss(self, X, y=None):
		"""
		Compute loss and gradient for the fully-connected net.

		Input / output: Same as TwoLayerNet above.
		"""
		X = X.astype(self.dtype)
		mode = 'test' if y is None else 'train'

		# Set train/test mode for batchnorm params and dropout param since they
		# behave differently during training and testing.
		if self.use_dropout:
			self.dropout_param['mode'] = mode
		if self.normalization=='batchnorm':
			for bn_param in self.bn_params:
				bn_param['mode'] = mode
		scores = None
		H = self.num_layers - 1

		cache = {}
		reg= self.reg
		reg_loss = 0


		################################## forward pass ####################################

		xi = X
		for i in range(self.num_layers):
			ith_str = str(i+1) #pick 1 bigger as string in keys of params dict
			wi = self.params['W' + ith_str]
			bi = self.params['b' + ith_str]
			if reg is not 0: reg_loss += np.sum(wi * wi)

			last_layer = i == H
			if last_layer:  # next will be softmax not relu
				xi, cache[H] = affine_forward(xi, wi, bi)
			else:
				if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
					gamma = self.params['gamma'+ith_str]
					beta = self.params['beta'+ith_str]
					xi, cache[i] = ln_bn_forward(xi, wi, bi, gamma, beta, self.bn_params[i], self.normalization)

				elif self.normalization == 'somethingelse':
					pass
				else:
					xi, cache[i] = affine_relu_forward(xi, wi, bi)

		scores = xi

		############################################################################
		# TODO: Implement the forward pass for the fully-connected net, computing  #
		# the class scores for X and storing them in the scores variable.          #
		#                                                                          #
		# When using dropout, you'll need to pass self.dropout_param to each       #
		# dropout forward pass.                                                    #
		#                                                                          #
		# When using batch normalization, you'll need to pass self.bn_params[0] to #
		# the forward pass for the first batch normalization layer, pass           #
		# self.bn_params[1] to the forward pass for the second batch normalization #
		# layer, etc.                                                              #
		############################################################################

		# If test mode return early
		if mode == 'test':
			return scores

		loss, grads = 0.0, {}

		num_loss, dscores = softmax_loss(scores, y)
		loss = num_loss + 0.5*reg*reg_loss

		################################## backward pass ####################################

		dx = dscores

		for i in range(self.num_layers):
			ith_str = str(self.num_layers-i)
			wi = self.params['W' + ith_str]
			if i == 0:
				#for last layer
				# print(cache[H][1])
				dx, dwi, dbi = affine_backward(dx, cache[H])
				grads['W' + ith_str] = dwi + reg * wi
				grads['b' + ith_str] = dbi
			else:
				if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
					dx, dwi, dbi, dgamma, dbeta = ln_bn_backward(dx, cache[H-i], self.normalization)
					grads['gamma' + ith_str] = dgamma
					grads['beta' + ith_str] = dbeta
					grads['W' + ith_str] = dwi + reg * wi
					grads['b' + ith_str] = dbi

				elif self.normalization == "somethingesle":
					pass
				else:
					dx, dwi, dbi = affine_relu_backward(dx, cache[H-i])
					grads['W' + ith_str] = dwi + reg * wi
					grads['b' + ith_str] = dbi


		############################################################################
		# TODO: Implement the backward pass for the fully-connected net. Store the #
		# loss in the loss variable and gradients in the grads dictionary. Compute #
		# data loss using softmax, and make sure that grads[k] holds the gradients #
		# for self.params[k]. Don't forget to add L2 regularization!               #
		#                                                                          #
		# When using batch/layer normalization, you don't need to regularize the scale   #
		# and shift parameters.                                                    #
		#                                                                          #
		# NOTE: To ensure that your implementation matches ours and you pass the   #
		# automated tests, make sure that your L2 regularization includes a factor #
		# of 0.5 to simplify the expression for the gradient.                      #
		############################################################################
		pass
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		return loss, grads
