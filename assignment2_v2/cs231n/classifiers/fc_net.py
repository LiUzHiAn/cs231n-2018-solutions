from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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

	def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
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

		############################################################################
		# TODO: Initialize the weights and biases of the two-layer net. Weights    #
		# should be initialized from a Gaussian centered at 0.0 with               #
		# standard deviation equal to weight_scale, and biases should be           #
		# initialized to zero. All weights and biases should be stored in the      #
		# dictionary self.params, with first layer weights                         #
		# and biases using the keys 'W1' and 'b1' and second layer                 #
		# weights and biases using the keys 'W2' and 'b2'.                         #
		############################################################################
		W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
		b1 = np.zeros(hidden_dim)
		W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
		b2 = np.zeros(num_classes)

		self.params["W1"] = W1
		self.params["W2"] = W2
		self.params["b1"] = b1
		self.params["b2"] = b2

	############################################################################
	#                             END OF YOUR CODE                             #
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
		############################################################################
		# TODO: Implement the forward pass for the two-layer net, computing the    #
		# class scores for X and storing them in the scores variable.              #
		############################################################################
		W1 = self.params["W1"]
		W2 = self.params["W2"]
		b1 = self.params["b1"]
		b2 = self.params["b2"]
		# 第一隐藏层输出
		(H1, cache1) = affine_relu_forward(X, W1, b1)
		(scores, cache2) = affine_forward(H1, W2, b2)
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		# If y is None then we are in test mode so just return scores
		if y is None:
			return scores

		loss, grads = 0, {}
		############################################################################
		# TODO: Implement the backward pass for the two-layer net. Store the loss  #
		# in the loss variable and gradients in the grads dictionary. Compute data #
		# loss using softmax, and make sure that grads[k] holds the gradients for  #
		# self.params[k]. Don't forget to add L2 regularization!                   #
		#                                                                          #
		# NOTE: To ensure that your implementation matches ours and you pass the   #
		# automated tests, make sure that your L2 regularization includes a factor #
		# of 0.5 to simplify the expression for the gradient.                      #
		############################################################################
		(loss, dscores) = softmax_loss(scores, y)
		# 正则化参数部分产生的loss
		reg_loss = 0.5 * self.reg * np.sum(np.square(W1)) + 0.5 * self.reg * np.sum(np.square(W2))
		loss += reg_loss
		# print(loss)
		(dH1, dW2, db2) = affine_backward(dscores, cache2)
		(dX, dW1, db1) = affine_relu_backward(dH1, cache1)

		# 一定别忘了加上正则项那部分产生的dL/dW
		dW1 += self.reg * W1
		dW2 += self.reg * W2
		grads["W1"] = dW1
		grads["W2"] = dW2
		grads["b1"] = db1
		grads["b2"] = db2
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

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

	def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
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

		self.input_dim = input_dim

		############################################################################
		# TODO: Initialize the parameters of the network, storing all values in    #
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
		all_layers_dims = [input_dim] + hidden_dims + [num_classes]
		print(all_layers_dims)
		for i in range(1, self.num_layers + 1):
			self.params["W{}".format(i)] = np.random.randn(all_layers_dims[i - 1],
														   all_layers_dims[i]) * weight_scale
			self.params["b{}".format(i)] = np.zeros(all_layers_dims[i])

		# 如果采用BN
		if self.normalization == "batchnorm":
			for i in range(1, self.num_layers):
				self.params["gamma{}".format(i)] = np.ones(all_layers_dims[i])
				self.params["beta{}".format(i)] = np.zeros(all_layers_dims[i])
		############################################################################
		#                             END OF YOUR CODE                             #
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
		if self.normalization == 'batchnorm':
			self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
		if self.normalization == 'layernorm':
			self.bn_params = [{} for i in range(self.num_layers - 1)]

		# Cast all parameters to the correct datatype
		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)

	def loss(self, X, y=None):
		"""
		Compute loss and gradient for the fully-connected net.

		Input / output: Same as TwoLayerNet above.
		"""
		# print("当前X的shape是", X.shape)
		X = X.astype(self.dtype)
		# !!!!!一定记住，loss()函数的输入X是(N,d1,d2...)形式的，一定要先转成(N,D)格式!!!!!
		X = np.reshape(X, newshape=(X.shape[0], self.input_dim))
		mode = 'test' if y is None else 'train'

		# Set train/test mode for batchnorm params and dropout param since they
		# behave differently during training and testing.
		if self.use_dropout:
			self.dropout_param['mode'] = mode
		if self.normalization == 'batchnorm':
			for bn_param in self.bn_params:
				bn_param['mode'] = mode
		scores = None
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
		cache_linear = {}
		cache_relu = {}
		cache_BN = {}
		BN_out = {}
		linear_out = {}
		relu_out = {}
		dropout_cache = {}

		# 这里只是为了循环里面方便而设置relu_out[0]，其实没有意义。
		# 我们只从第一层开始运用BN，Linear，Non-linear
		relu_out[0] = X

		# 最后一层不用relu，而是softmax，所以在num_layser-1上循环
		for i in range(1, self.num_layers):
			# print(i)

			W = self.params["W{}".format(i)]
			b = self.params["b{}".format(i)]

			# 先线性层
			(linear_out[i], cache_linear[i]) = affine_forward(relu_out[i - 1], W, b)
			# 如果采用BN
			if self.normalization == "batchnorm":
				gamma = self.params["gamma{}".format(i)]
				beta = self.params["beta{}".format(i)]
				# Batch Normalization
				(BN_out[i], cache_BN[i]) = batchnorm_forward(linear_out[i], gamma, beta, self.bn_params[i - 1])
			# 不用BN
			else:
				BN_out[i] = linear_out[i]
			# 最后non-linear层
			(relu_out[i], cache_relu[i]) = relu_forward(BN_out[i])
			# 如果使用dropout
			if self.use_dropout:
				relu_out[i], dropout_cache[i] = dropout_forward(relu_out[i], self.dropout_param)
		# 最后一层用softmax
		W = self.params["W{}".format(self.num_layers)]
		b = self.params["b{}".format(self.num_layers)]
		(scores, cache_last) = affine_forward(relu_out[self.num_layers - 1], W, b)

		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		# If test mode return early
		if mode == 'test':
			return scores

		loss, grads = 0.0, {}
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
		(loss, dscores) = softmax_loss(scores, y)
		# 正则化参数部分产生的loss
		for i in range(1, self.num_layers + 1):
			loss += 0.5 * self.reg * np.sum(np.square(self.params["W{}".format(i)]))

		# print(loss)

		"""----------------------反向传播-------------------------"""
		dout_BN = {}
		dout_linear = {}
		dout_relu = {}

		# 最后一层的BP略有不同
		(dout_relu[self.num_layers - 1], grads["W{}".format(self.num_layers)],
		 grads["b{}".format(self.num_layers)]) = affine_backward(
			dscores, cache_last)

		# 除最后一层外的BP
		for i in range(self.num_layers - 1, 0, -1):
			if self.use_dropout:
				dout_relu[i] = dropout_backward(dout_relu[i], dropout_cache[i])
			# 先relu层的BP
			dout_BN[i] = relu_backward(dout_relu[i], cache_relu[i])
			# 如果用了BN
			if self.normalization == "batchnorm":
				# BN层的BP
				(dout_linear[i], grads["gamma{}".format(i)], grads["beta{}".format(i)]) = batchnorm_backward_alt(
					dout_BN[i], cache_BN[i])
			# 没用BN
			else:
				dout_linear[i] = dout_BN[i]
			# 最后Linear层的BP
			(dout_relu[i - 1], grads["W{}".format(i)], grads["b{}".format(i)]) = affine_backward(dout_linear[i],
																								 cache_linear[i])
		# 一定别忘了加上正则项那部分产生的dL/dW
		for i in range(1, self.num_layers + 1):
			grads["W{}".format(i)] += self.reg * self.params["W{}".format(i)]
		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################

		return loss, grads
