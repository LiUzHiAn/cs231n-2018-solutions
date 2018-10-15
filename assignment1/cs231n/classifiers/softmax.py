import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
	"""
	Softmax loss function, naive implementation (with loops)

	Inputs have dimension D, there are C classes, and we operate on minibatches
	of N examples.

	Inputs:
	- W: A numpy array of shape (D, C) containing weights.
	- X: A numpy array of shape (N, D) containing a minibatch of data.
	- y: A numpy array of shape (N,) containing training labels; y[i] = c means
	  that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)
	# compute the loss and the gradient
	num_classes = W.shape[1]
	num_train = X.shape[0]
	for i in range(num_train):
		# 根据softmax公式得出
		f = np.dot(X[i], W)
		# 防止分子分母同时过大，先都减去最大值
		f -= np.max(f)
		loss += np.log(np.sum(np.exp(f))) - f[y[i]]

		# 求梯度
		exp_sum = np.sum(np.exp(f))
		dW[:, y[i]] -= X[i]
		for j in range(num_classes):
			dW[:, j] += (1 / exp_sum) * np.exp(f[j]) * X[i]

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using explicit loops.     #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	loss = loss / num_train + reg * np.sum(W * W)
	dW = dW / num_train + 2 * reg * W
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	"""
	Softmax loss function, vectorized version.

	Inputs and outputs are the same as softmax_loss_naive.
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)
	# compute the loss and the gradient
	num_train = X.shape[0]

	f = np.dot(X, W)
	# 防止分子分母同时过大，先都减去最大值
	f -= np.max(f, axis=1).reshape(num_train, 1)

	exp_sum = np.sum(np.exp(f), axis=1)
	L = -f[range(0, num_train), y] + np.log(exp_sum)
	loss = np.sum(L, axis=0) / num_train

	tmp = np.zeros_like(f)
	tmp[range(0, num_train), y] = -1
	# 这里运用了广播机制
	tmp += np.exp(f) / exp_sum.reshape(num_train, 1)
	dW = np.dot(X.T, tmp)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	loss += reg * np.sum(W * W)
	dW = dW / num_train + 2 * reg * W
	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW
