import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
	"""
	Structured SVM loss function, naive implementation (with loops).

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
	dW = np.zeros(W.shape)  # initialize the gradient as zero

	# compute the loss and the gradient
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	for i in range(num_train):
		scores = X[i].dot(W)
		correct_class_score = scores[y[i]]
		for j in range(num_classes):
			if j == y[i]:
				continue
			margin = scores[j] - correct_class_score + 1  # note delta = 1
			if margin > 0:
				loss += margin
				# L_i=max(0,s_j-s_y_i+delta)
				dW[:, j] += X[i]
				dW[:, y[i]] -= X[i]

	# Right now the loss is a sum over all training examples, but we want it
	# to be an average instead so we divide by num_train.
	loss /= num_train
	dW /= num_train

	# Add regularization to the loss.
	loss += reg * np.sum(W * W)
	dW += 2 * reg * W

	#############################################################################
	# TODO:                                                                     #
	# Compute the gradient of the loss function and store it dW.                #
	# Rather that first computing the loss and then computing the derivative,   #
	# it may be simpler to compute the derivative at the same time that the     #
	# loss is being computed. As a result you may need to modify some of the    #
	# code above to compute the gradient.                                       #
	#############################################################################

	return loss, dW


def svm_loss_vectorized(W, X, y, reg):
	"""
	Structured SVM loss function, vectorized implementation.

	Inputs and outputs are the same as svm_loss_naive.
	"""
	loss = 0.0
	dW = np.zeros(W.shape)  # initialize the gradient as zero

	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the structured SVM loss, storing the    #
	# result in loss.                                                           #
	#############################################################################
	# compute the loss and the gradient
	num_train = X.shape[0]

	scores = np.dot(X, W)
	# correct_score每行对应每个样本的正确分类对应的得分,reshape的目的是为了broadcast
	correct_socore = scores[range(0, num_train), y].reshape(num_train, 1)
	# 1是SVM的delta
	L = scores - correct_socore + 1
	# 正确的分类不要加loss
	L[range(0, num_train), y] = 0
	# 等同于max函数
	L[L < 0] = 0
	# 所有loss_ij求和
	loss = np.sum(L)

	loss /= num_train
	loss += reg * np.sum(W * W)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the gradient for the structured SVM     #
	# loss, storing the result in dW.                                           #
	#                                                                           #
	# Hint: Instead of computing the gradient from scratch, it may be easier    #
	# to reuse some of the intermediate values that you used to compute the     #
	# loss.                                                                     #
	#############################################################################
	tmp = np.zeros(L.shape)
	tmp[L > 0] = 1
	# 根据公式L_i=max(0,s_j-s_y_i+delta)
	# 对每个训练样本对应的score向量，y_i对应的score会在j循环中不断加，而s_j只会加一次
	# 也就是说，dL_i/ds在j循环时不断加-X[i]，而只会加X[i]一次

	# 统计要减多少次
	row_sum = np.sum(tmp, axis=1)
	tmp[range(0, num_train), y] = -row_sum

	dW = np.dot(X.T, tmp) / num_train
	dW += 2 * reg * W

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return loss, dW
