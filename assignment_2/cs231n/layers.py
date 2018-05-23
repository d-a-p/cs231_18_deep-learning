from builtins import range
import numpy as np


def affine_forward(x, w, b):
	"""
	Computes the forward pass for an affine (fully-connected) layer.

	The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
	examples, where each example x[i] has shape (d_1, ..., d_k). We will
	reshape each input into a vector of dimension D = d_1 * ... * d_k, and
	then transform it to an output vector of dimension M.

	Inputs:
	- x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
	- w: A numpy array of weights, of shape (D, M)
	- b: A numpy array of biases, of shape (M,)

	Returns a tuple of:
	- out: output, of shape (N, M)
	- cache: (x, w, b)
	"""
	N = x.shape[0]
	x_processed = np.reshape(x, (N,-1))
	out = x_processed.dot(w) + b
	###########################################################################
	# Implement the affine forward pass. Store the result in out. You         #
	# will need to reshape the input into rows.                               #
	###########################################################################
	cache = (x, w, b)
	return out, cache


def affine_backward(dout, cache):
	"""
	Computes the backward pass for an affine layer.

	Inputs:
	- dout: Upstream derivative, of shape (N, M)
	- cache: Tuple of:
	  - x: Input data, of shape (N, d_1, ... d_k)
	  - w: Weights, of shape (D, M)
	  - b: Biases, of shape (M,)

	Returns a tuple of:
	- dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
	- dw: Gradient with respect to w, of shape (D, M)
	- db: Gradient with respect to b, of shape (M,)
	"""
	x, w, b = cache
	dx, dw, db = None, None, None
	N = x.shape[0]
	x_processed = np.reshape(x, (N,-1))
	###########################################################################
	#Implement the affine backward pass.                                      #
	###########################################################################
	dx_im = dout.dot(w.T)
	dx = np.reshape(dx_im, x.shape)
	dw = x_processed.T.dot(dout)
	db = np.sum(dout, axis=0)

	return dx, dw, db


def relu_forward(x):
	"""
	Computes the forward pass for a layer of rectified linear units (ReLUs).

	Input:
	- x: Inputs, of any shape

	Returns a tuple of:
	- out: Output, of the same shape as x
	- cache: x
	"""
	out = np.maximum(0, x)
	###########################################################################
	#Implement the ReLU forward pass.                                         #
	###########################################################################

	cache = x
	return out, cache


def relu_backward(dout, cache):
	"""
	Computes the backward pass for a layer of rectified linear units (ReLUs).

	Input:
	- dout: Upstream derivatives, of any shape
	- cache: Input x, of same shape as dout

	Returns:
	- dx: Gradient with respect to x
	"""
	dx, x = None, cache
	dx = dout*(x>0)
	###########################################################################
	# Implement the ReLU backward pass.                                       #
	###########################################################################
	return dx


def batchnorm_forward(x, gamma, beta, bn_param):
	"""
	Forward pass for batch normalization.

	During training the sample mean and (uncorrected) sample variance are
	computed from minibatch statistics and used to normalize the incoming data.
	During training we also keep an exponentially decaying running mean of the
	mean and variance of each feature, and these averages are used to normalize
	data at test-time.

	At each timestep we update the running averages for mean and variance using
	an exponential decay based on the momentum parameter:

	running_mean = momentum * running_mean + (1 - momentum) * sample_mean
	running_var = momentum * running_var + (1 - momentum) * sample_var

	Note that the batch normalization paper suggests a different test-time
	behavior: they compute sample mean and variance for each feature using a
	large number of training images rather than using a running average. For
	this implementation we have chosen to use running averages instead since
	they do not require an additional estimation step; the torch7
	implementation of batch normalization also uses running averages.

	Input:
	- x: Data of shape (N, D)
	- gamma: Scale parameter of shape (D,)
	- beta: Shift paremeter of shape (D,)
	- bn_param: Dictionary with the following keys:
	  - mode: 'train' or 'test'; required
	  - eps: Constant for numeric stability
	  - momentum: Constant for running mean / variance.
	  - running_mean: Array of shape (D,) giving running mean of features
	  - running_var Array of shape (D,) giving running variance of features

	Returns a tuple of:
	- out: of shape (N, D)
	- cache: A tuple of values needed in the backward pass
	"""
	mode = bn_param['mode']
	eps = bn_param.get('eps', 1e-5)
	momentum = bn_param.get('momentum', 0.9)

	N, D = x.shape
	running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
	running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

	out, cache = None, None
	if mode == 'train':
		u_x = np.mean(x, axis=0)
		sigma = np.std(x, axis=0)
		var = sigma**2
		x_tilda = (x - u_x)/np.sqrt(var + eps)
		y_tilda = gamma * x_tilda + beta
		out = y_tilda

		cache = (x_tilda, beta, gamma, eps, x, u_x, var)

		running_mean = momentum * running_mean + (1 - momentum) * u_x
		running_var = momentum * running_var + (1 - momentum) * var
		#######################################################################
		# Implement the training-time forward pass for batch norm.            #
		# Use minibatch statistics to compute the mean and variance, use      #
		# these sprint(len(cache[i]))tatistics to normalize the incoming data, and scale and      #
		# shift the normalized data using gamma and beta.                     #
		#                                                                     #
		# You should store the output in the variable out. Any intermediates  #
		# that you need for the backward pass should be stored in the cache   #
		# variable.                                                           #
		#                                                                     #
		# You should also use your computed sample mean and variance together #
		# with the momentum variable to update the running mean and running   #
		# variance, storing your result in the running_mean and running_var   #
		# variables.                                                          #
		#                                                                     #
		# Note that though you should be keeping track of the running         #
		# variance, you should normalize the data based on the standard       #
		# deviation (square root of variance) instead!                        #
		# Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
		# might prove to be helpful.                                          #
		#######################################################################

	elif mode == 'test':
		u_x = running_mean
		var = running_var
		x_tilda = (x - u_x)/np.sqrt(var + eps)
		y_tilda = gamma * x_tilda + beta

		out = y_tilda
		cache = (x_tilda, beta, gamma, eps, x, u_x, var)
		#######################################################################
		# Implement the test-time forward pass for batch normalization.       #
		# Use the running mean and variance to normalize the incoming data,   #
		# then scale and shift the normalized data using gamma and beta.      #
		# Store the result in the out variable.                               #
		#######################################################################

	else:
		raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

	# Store the updated running means back into bn_param
	bn_param['running_mean'] = running_mean
	bn_param['running_var'] = running_var

	return out, cache


def batchnorm_backward(dout, cache):
	"""
	Backward pass for batch normalization.

	For this implementation, you should write out a computation graph for
	batch normalization on paper and propagate gradients backward through
	intermediate nodes.

	Inputs:
	- dout: Upstream derivatives, of shape (N, D)
	- cache: Variable of intermediates from batchnorm_forward.

	Returns a tuple of:
	- dx: Gradient with respect to inputs x, of shape (N, D)
	- dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
	- dbeta: Gradient with respect to shift parameter beta, of shape (D,)
	"""
	dx, dgamma, dbeta = None, None, None

	x_tilda, beta, gamma, eps, x, u_x, var = cache

	dbeta = np.sum(dout, axis=0)
	dgamma = np.sum(dout * x_tilda, axis=0)
   
	d_xhat = dout * gamma
	sqrt_var = np.sqrt(var + eps)
	inv_var = 1./ sqrt_var
	x_mu = x - u_x
	N = x.shape[0]

	dvar = -0.5 * np.sum(d_xhat * x_mu, axis = 0) * inv_var**3
	du = -1. * np.sum(d_xhat, axis =0) * inv_var + dvar * -2. / N * np.sum(x_mu, axis=0)
	dx = d_xhat * inv_var + (dvar * x_mu * 2. + du)/ N

	###########################################################################
	# Implement the backward pass for batch normalization. Store the          #
	# results in the dx, dgamma, and dbeta variables.                         #
	# Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
	# might prove to be helpful.                                              #
	###########################################################################

	return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
	#http://cthorey.github.io./backpropagation/
	"""
	Alternative backward pass for batch normalization.

	For this implementation you should work out the derivatives for the batch
	normalizaton backward pass on paper and simplify as much as possible. You
	should be able to derive a simple expression for the backward pass.
	See the jupyter notebook for more hints.

	Note: This implementation should expect to receive the same cache variable
	as batchnorm_backward, but might not use all of the values in the cache.

	Inputs / outputs: Same as batchnorm_backward
	"""
	dx, dgamma, dbeta = None, None, None
	x_tilda, beta, gamma, eps, x, u_x, var = cache
	N = x.shape[0]
	x_hat = x_tilda
	dxhat = dout * gamma

	#https://kevinzakka.github.io/2016/09/14/batch_normalization/

	dbeta = np.sum(dout, axis=0)
	dgamma = np.sum(x_hat*dout, axis=0)
	dx = (N * dxhat - np.sum(dxhat, axis=0) - x_hat * np.sum(dxhat * x_hat, axis=0)) / (N * np.sqrt(var + eps))

	###########################################################################
	# Implement the backward pass for batch normalization. Store the          #
	# results in the dx, dgamma, and dbeta variables.                         #
	#                                                                         #
	# After computing the gradient with respect to the centered inputs, you   #
	# should be able to compute gradients with respect to the inputs in a     #
	# single statement; our implementation fits on a single 80-character line.#
	###########################################################################
  
	return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
	"""
	Forward pass for layer normalization.

	During both training and test-time, the incoming data is normalized per data-point,
	before being scaled by gamma and beta parameters identical to that of batch normalization.

	Note that in contrast to batch normalization, the behavior during train and test-time for
	layer normalization are identical, and we do not need to keep track of running averages
	of any sort.

	Input:
	- x: Data of shape (N, D)
	- gamma: Scale parameter of shape (D,)
	- beta: Shift paremeter of shape (D,)
	- ln_param: Dictionary with the following keys:
		- eps: Constant for numeric stability

	Returns a tuple of:
	- out: of shape (N, D)
	- cache: A tuple of values needed in the backward pass
	"""
	out, cache = None, None
	eps = ln_param.get('eps', 1e-5)
	x = x.T

	u_x = np.mean(x, axis=0)
	sigma = np.std(x, axis=0)
	var = sigma**2
	x_tilda = (x - u_x)/np.sqrt(var + eps)
	x_tilda_t = x_tilda.T
	y_tilda = gamma * x_tilda_t + beta
	out = y_tilda

	cache = (x_tilda_t, beta, gamma, eps, x, u_x, var)
	###########################################################################
	# Implement the training-time forward pass for layer norm.                #
	# Normalize the incoming data, and scale and  shift the normalized data   #
	#  using gamma and beta.                                                  #
	# HINT: this can be done by slightly modifying your training-time         #
	# implementation of  batch normalization, and inserting a line or two of  #
	# well-placed code. In particular, can you think of any matrix            #
	# transformations you could perform, that would enable you to copy over   #
	# the batch norm code and leave it almost unchanged?                      #
	###########################################################################

	return out, cache


def layernorm_backward(dout, cache):
	"""
	Backward pass for layer normalization.

	For this implementation, you can heavily rely on the work you've done already
	for batch normalization.

	Inputs:
	- dout: Upstream derivatives, of shape (N, D)
	- cache: Variable of intermediates from layernorm_forward.

	Returns a tuple of:
	- dx: Gradient with respect to inputs x, of shape (N, D)
	- dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
	- dbeta: Gradient with respect to shift parameter beta, of shape (D,)
	"""
	dx, dgamma, dbeta = None, None, None

	x_tilda, beta, gamma, eps, x, u_x, var = cache
	dbeta = np.sum(dout, axis=0)
	dgamma = np.sum(dout * x_tilda, axis=0)

	d_xhat = dout * gamma #N, D
	d_xhat_t = d_xhat.T #N, D
	sqrt_var = np.sqrt(var + eps) #D
	inv_var = 1. / sqrt_var #D
	x_mu = x - u_x #D,N
	D = x.shape[0]

	dvar = -0.5 * np.sum(d_xhat_t * x_mu, axis=0) * inv_var ** 3 #D
	du = -1. * np.sum(d_xhat_t, axis=0) * inv_var + dvar * -2. / D * np.sum(x_mu, axis=0) #D
	dx = d_xhat_t * inv_var + (dvar * x_mu * 2. + du) / D
	dx = dx.T
	###########################################################################
	# Implement the backward pass for layer norm.                             #
	#                                                                         #
	# HINT: this can be done by slightly modifying your training-time         #
	# implementation of batch normalization. The hints to the forward pass    #
	# still apply!                                                            #
	###########################################################################
	return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
	"""
	Performs the forward pass for (inverted) dropout.

	Inputs:
	- x: Input data, of any shape
	- dropout_param: A dictionary with the following keys:
	  - p: Dropout parameter. We keep each neuron output with probability p.
	  - mode: 'test' or 'train'. If the mode is train, then perform dropout;
		if the mode is test, then just return the input.
	  - seed: Seed for the random number generator. Passing seed makes this
		function deterministic, which is needed for gradient checking but not
		in real networks.

	Outputs:
	- out: Array of the same shape as x.
	- cache: tuple (dropout_param, mask). In training mode, mask is the dropout
	  mask that was used to multiply the input; in test mode, mask is None.

	NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
	See http://cs231n.github.io/neural-networks-2/#reg for more details.

	NOTE 2: Keep in mind that p is the probability of **keep** a neuron
	output; this might be contrary to some sources, where it is referred to
	as the probability of dropping a neuron output.
	"""
	p, mode = dropout_param['p'], dropout_param['mode']
	if 'seed' in dropout_param:
		np.random.seed(dropout_param['seed'])

	mask = None
	out = None

	if mode == 'train':

		mask = (np.random.rand(*x.shape) < p) / p
		out = x * mask
		#######################################################################
		# Implement training phase forward pass for inverted dropout.         #
		# Store the dropout mask in the mask variable.                        #
		#######################################################################

	elif mode == 'test':
		out = x
		mask = None
		#######################################################################
		# Implement the test phase forward pass for inverted dropout.         #
		#######################################################################

	cache = (dropout_param, mask)
	out = out.astype(x.dtype, copy=False)

	return out, cache


def dropout_backward(dout, cache):
	"""
	Perform the backward pass for (inverted) dropout.

	Inputs:
	- dout: Upstream derivatives, of any shape
	- cache: (dropout_param, mask) from dropout_forward.
	"""
	dropout_param, mask = cache
	mode = dropout_param['mode']

	dx = None
	if mode == 'train':
		dx = dout * mask
		#######################################################################
		# Implement training phase backward pass for inverted dropout         #
		#######################################################################

	elif mode == 'test':
		dx = dout
	return dx


def conv_forward_naive(x, w, b, conv_param):
	"""
	A naive implementation of the forward pass for a convolutional layer.

	The input consists of N data points, each with C channels, height H and
	width W. We convolve each input with F different filters, where each filter
	spans all C channels and has height HH and width WW.

	Input:
	- x: Input data of shape (N, C, H, W)
	- w: Filter weights of shape (F, C, HH, WW)
	- b: Biases, of shape (F,)
	- conv_param: A dictionary with the following keys:
	  - 'stride': The number of pixels between adjacent receptive fields in the
		horizontal and vertical directions.
	  - 'pad': The number of pixels that will be used to zero-pad the input.


	During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
	along the height and width axes of the input. Be careful not to modfiy the original
	input x directly.

	Returns a tuple of:
	- out: Output data, of shape (N, F, H', W') where H' and W' are given by
	  H' = 1 + (H + 2 * pad - HH) / stride
	  W' = 1 + (W + 2 * pad - WW) / stride
	- cache: (x, w, b, conv_param)
	"""
	out = None

	stride = conv_param.get('stride', 1)
	pad = conv_param.get('pad', 0)

	N, F = x.shape[0], w.shape[0]
	HH, H = w.shape[2], x.shape[2]
	WW, W = w.shape[3], x.shape[3]

	Ho = int(1 + (H + 2 * pad - HH) / stride)
	Wo = int(1 + (W + 2 * pad - WW) / stride)
	V = np.zeros((N, F, Ho, Wo))

	npad = ((0, 0), (0, 0), (pad, pad), (pad, pad))
	x_padded = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

	for i in range(N):
		x_i = x_padded[i]
		for j in range(F):
			w_j = w[j]
			b_j = b[j]
			for m in range(Ho):
				h_from = m * stride
				h_to = h_from + HH
				for n in range(Wo):
					w_from = n * stride
					w_to = w_from + WW
					roi = x_i[:,h_from:h_to,w_from:w_to]
					V[i, j, m, n] = np.sum(roi * w_j) + b_j

	out = V
	###########################################################################
	# Implement the convolutional forward pass.                               #
	# Hint: you can use the function np.pad for padding.                      #
	###########################################################################

	cache = (x, w, b, conv_param)
	return out, cache


def conv_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a convolutional layer.

	Inputs:
	- dout: Upstream derivatives.   # (N, F, Ho, Wo)
	- cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive


	Returns a tuple of:
	- dx: Gradient with respect to x  # (N, C, H, W)
	- dw: Gradient with respect to w  # (F, C, HH, WW)
	- db: Gradient with respect to b  # F
	"""

	dx, dw, db = None, None, None

	x, w, b, conv_param = cache

	# https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e

	stride = conv_param.get('stride', 1)
	pad = conv_param.get('pad', 0)

	C = x.shape[1]

	HH, H = w.shape[2], x.shape[2]
	WW, W = w.shape[3], x.shape[3]
	N, F, Ho, Wo = dout.shape

	npad = ((0, 0), (0, 0), (pad, pad), (pad, pad))
	x_padded = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
	dW = np.zeros((F, C, HH, WW))
	dB = np.zeros((F))

	for i in range(C):
		x_i = x_padded[:, i]    # (N, H, W)
		for j in range(F):
			d_j = dout[:, j]   # (N, Ho, Wo)
			dB[j] = np.sum(dout[:, j])
			for m in range(HH):
				h_from = m * stride
				h_to = h_from + Ho
				for n in range(WW):
					w_from = n * stride
					w_to = w_from + Wo
					roi = x_i[:,h_from:h_to,w_from:w_to]
					dW[j, i, m, n] = np.sum(roi * d_j)

	dw = dW
	db = dB

	# from : H = 1 + (H + 2 * pad - HH) / stride
	#special case for H =31, HH = 3, dh= 16 so required pad = 17

	required_pad = ((H-1)*stride - Ho + HH)
	x_pad = int(np.floor(required_pad/2))
	y_pad = int(np.ceil(required_pad/2))
	npad = ((0, 0), (0, 0), (x_pad, y_pad), (x_pad, y_pad))
	d_padded = np.pad(dout, pad_width=npad, mode='constant', constant_values=0)

	dX = np.zeros((N, C, H, W))
	w_fliped = w[:,:,::-1,::-1]

	for i in range(N):
		d_i = d_padded[i]    # (F, Ho, Wo)
		for j in range(C):
			w_j = w_fliped[:, j]    # (F, HH, WW)
			for m in range(H):
				h_from = m * stride
				h_to = h_from + HH
				for n in range(W):
					w_from = n * stride
					w_to = w_from + WW
					roi = d_i[:,h_from:h_to,w_from:w_to]
					dX[i, j, m, n] = np.sum(roi * w_j)

	dx = dX

	# integrated with for loop of dW
	# dB = np.zeros((F))
	# for f in range(F):
	# 	dB[f] = np.sum(dout[:, f])
	# db = dB

	# ###########################################################################
	# Implement the convolutional backward pass.                              #
	###########################################################################

	return dx, dw, db


def max_pool_forward_naive(x, pool_param):
	"""
	A naive implementation of the forward pass for a max-pooling layer.

	Inputs:
	- x: Input data, of shape (N, C, H, W)
	- pool_param: dictionary with the following keys:
	  - 'pool_height': The height of each pooling region
	  - 'pool_width': The width of each pooling region
	  - 'stride': The distance between adjacent pooling regions

	No padding is necessary here. Output size is given by

	Returns a tuple of:
	- out: Output data, of shape (N, C, H', W') where H' and W' are given by
	  H' = 1 + (H - pool_height) / stride
	  W' = 1 + (W - pool_width) / stride
	- cache: (x, pool_param)
	"""
	out = None

	# default is 2*2 pool filter with 2 stride
	ph = pool_param.get('pool_height', 2)
	pw = pool_param.get('pool_width', 2)
	ps = pool_param.get('stride', 2)

	N, C, H, W = x.shape

	Ho = int(1 + (H - ph) / ps)
	Wo = int(1 + (W - pw) / ps)

	PL = np.zeros((N, C, Ho, Wo))

	for i in range(N):
		for j in range(C):
			x_i = x[i, j]       # H, W
			for m in range(Ho):
				h_from = m * ps
				h_to = h_from + ph
				for n in range(Wo):
					w_from = n * ps
					w_to = w_from + pw
					PL[i, j, m, n] = np.max(x_i[h_from:h_to, w_from:w_to])

	out = PL

	###########################################################################
	# Implement the max-pooling forward pass                                  #
	###########################################################################

	cache = (x, pool_param)
	return out, cache


def max_pool_backward_naive(dout, cache):
	"""
	A naive implementation of the backward pass for a max-pooling layer.

	Inputs:
	- dout: Upstream derivatives
	- cache: A tuple of (x, pool_param) as in the forward pass.

	Returns:
	- dx: Gradient with respect to x
	"""
	dx = None

	x, pool_param = cache

	ph = pool_param.get('pool_height', 2)
	pw = pool_param.get('pool_width', 2)
	ps = pool_param.get('stride', 2)

	N, C, dh, dw = dout.shape

	dX = np.zeros_like(x)

	for i in range(N):
		for j in range(C):
			x_i = x[i, j]       # H, W
			d_oi = dout[i,j]    #dh, dw
			for m in range(dh):
				h_from = m * ps
				h_to = h_from + ph
				for n in range(dw):
					w_from = n * ps
					w_to = w_from + pw
					mask = np.ones((ph, pw))
					roi = x_i[h_from:h_to, w_from:w_to]
					mask[roi < np.max(roi)] = 0
					dX[i, j, h_from:h_to, w_from:w_to] = mask * d_oi[m, n]


	dx = dX

	###########################################################################
	# Implement the max-pooling backward pass                                 #
	###########################################################################

	return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
	"""
	Computes the forward pass for spatial batch normalization.

	Inputs:
	- x: Input data of shape (N, C, H, W)
	- gamma: Scale parameter, of shape (C,)
	- beta: Shift parameter, of shape (C,)
	- bn_param: Dictionary with the following keys:
	  - mode: 'train' or 'test'; required
	  - eps: Constant for numeric stability
	  - momentum: Constant for running mean / variance. momentum=0 means that
		old information is discarded completely at every time step, while
		momentum=1 means that new information is never incorporated. The
		default of momentum=0.9 should work well in most situations.
	  - running_mean: Array of shape (D,) giving running mean of features
	  - running_var Array of shape (D,) giving running variance of features

	Returns a tuple of:
	- out: Output data, of shape (N, C, H, W)
	- cache: Values needed for the backward pass
	"""
	out, cache = None, None

	N, C, H, W = x.shape

	# x_t = np.transpose(x, (1,0,2,3))
	x_processed = x.reshape((C, -1))
	y, cache = batchnorm_forward(x_processed.T, gamma, beta, bn_param)
	y_processed = y.T.reshape((C, N, H, W))
	out = np.transpose(y_processed,(1,0,2,3))
	###########################################################################
	# Implement the forward pass for spatial batch normalization.             #
	#                                                                         #
	# HINT: You can implement spatial batch normalization by calling the      #
	# vanilla version of batch normalization you implemented above.           #
	# Your implementation should be very short; ours is less than five lines. #
	###########################################################################


	return out, cache


def spatial_batchnorm_backward(dout, cache):
	"""
	Computes the backward pass for spatial batch normalization.

	Inputs:
	- dout: Upstream derivatives, of shape (N, C, H, W)
	- cache: Values from the forward pass

	Returns a tuple of:
	- dx: Gradient with respect to inputs, of shape (N, C, H, W)
	- dgamma: Gradient with respect to scale parameter, of shape (C,)
	- dbeta: Gradient with respect to shift parameter, of shape (C,)
	"""
	dx, dgamma, dbeta = None, None, None

	N, C, H, W = dout.shape
	dt = np.transpose(dout, (1, 0, 2, 3))
	d_processed = dt.reshape((C, -1))
	dx1, dgamma, dbeta = batchnorm_backward_alt(d_processed.T, cache)
	dx = dx1.T.reshape((N, C, H, W))
	###########################################################################
	# Implement the backward pass for spatial batch normalization.            #
	#                                                                         #
	# HINT: You can implement spatial batch normalization by calling the      #
	# vanilla version of batch normalization you implemented above.           #
	# Your implementation should be very short; ours is less than five lines. #
	###########################################################################

	return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
	"""
	Computes the forward pass for spatial group normalization.
	In contrast to layer normalization, group normalization splits each entry
	in the data into G contiguous pieces, which it then normalizes independently.
	Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

	Inputs:
	- x: Input data of shape (N, C, H, W)
	- gamma: Scale parameter, of shape (C,)
	- beta: Shift parameter, of shape (C,)
	- G: Integer mumber of groups to split into, should be a divisor of C
	- gn_param: Dictionary with the following keys:
	  - eps: Constant for numeric stability

	Returns a tuple of:
	- out: Output data, of shape (N, C, H, W)
	- cache: Values needed for the backward pass
	"""
	out, cache = None, None
	eps = gn_param.get('eps',1e-5)
	N, C, H, W = x.shape
	b = np.ones((N,C,H,W))
	be = b*beta
	ga = b*gamma
	nCG = C//G

	b = be.reshape((N*G, H*W*nCG)).T
	g = ga.reshape((N*G, H*W*nCG)).T
	x_processed = x.reshape((N*G, H*W*nCG)).T

	#ref: assignment_2/normalization.png
	# x_processed logic :
	#           x(N,C,H,W) -> x_intermediate(N, G, H*W*nCG) -> x_processed(N*G, H*W*nCG)

	y, cache = batchnorm_forward(x_processed, g, b, {'mode':'train', 'eps':eps, 'momentum':0})
	out = y.T.reshape((N,C,H,W))

	###########################################################################
	# Implement the forward pass for spatial group normalization.             #
	# This will be extremely similar to the layer norm implementation.        #
	# In particular, think about how you could transform the matrix so that   #
	# the bulk of the code is similar to both train-time batch normalization  #
	# and layer normalization!                                                #
	###########################################################################
	cache = (cache, G, beta, gamma)
	return out, cache


def spatial_groupnorm_backward(dout, cache):
	"""
	Computes the backward pass for spatial group normalization.

	Inputs:
	- dout: Upstream derivatives, of shape (N, C, H, W)
	- cache: Values from the forward pass

	Returns a tuple of:
	- dx: Gradient with respect to inputs, of shape (N, C, H, W)
	- dgamma: Gradient with respect to scale parameter, of shape (C,)
	- dbeta: Gradient with respect to shift parameter, of shape (C,)
	"""
	dx, dgamma, dbeta = None, None, None

	N, C, H, W = dout.shape
	b_cache, G, beta, gamma = cache
	x_tilda, beta, gamma, eps, x, u_x, var = b_cache
	nCG = C // G

	d_processed = dout.reshape((N * G, H * W * nCG)).T
	dX, _, _ = batchnorm_backward_alt(d_processed, b_cache)
	dx = dX.T.reshape((N, C, H, W))
	#can't use dbeta and dgamma from batchnorm_backward since it is suming over whole dout

	x_hat = x_tilda.T.reshape(dout.shape)
	dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)
	dgamma = np.sum(x_hat * dout, axis=(0,2,3), keepdims=True)

	###########################################################################
	# Implement the backward pass for spatial group normalization.            #
	# This will be extremely similar to the layer norm implementation.        #
	###########################################################################

	return dx, dgamma, dbeta


def svm_loss(x, y):
	"""
	Computes the loss and gradient using for multiclass SVM classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth
	  class for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
	  0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	N = x.shape[0]
	correct_class_scores = x[np.arange(N), y]
	margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
	margins[np.arange(N), y] = 0
	loss = np.sum(margins) / N
	num_pos = np.sum(margins > 0, axis=1)
	dx = np.zeros_like(x)
	dx[margins > 0] = 1
	dx[np.arange(N), y] -= num_pos
	dx /= N
	return loss, dx


def softmax_loss(x, y):
	"""
	Computes the loss and gradient for softmax classification.

	Inputs:
	- x: Input data, of shape (N, C) where x[i, j] is the score for the jth
	  class for the ith input.
	- y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
	  0 <= y[i] < C

	Returns a tuple of:
	- loss: Scalar giving the loss
	- dx: Gradient of the loss with respect to x
	"""
	shifted_logits = x - np.max(x, axis=1, keepdims=True)
	Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
	log_probs = shifted_logits - np.log(Z)
	probs = np.exp(log_probs)
	N = x.shape[0]
	loss = -np.sum(log_probs[np.arange(N), y]) / N
	dx = probs.copy()
	dx[np.arange(N), y] -= 1
	dx /= N
	return loss, dx
