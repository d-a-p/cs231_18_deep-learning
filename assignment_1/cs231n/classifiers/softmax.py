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
  print("W: ",W.shape)
  print("X: ",X.shape)
  print("y: ",y.shape)

  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  C = W.shape[1]

  scores = X.dot(W)
  print("scpores: ", scores.shape)

  p = np.zeros_like(scores)
  for i in range(scores.shape[0]):
      f = scores[i]  #1*C
      f -= np.max(f) # numaric stability
      exp_f = np.exp(f)
      sum = np.sum(exp_f)
      p = exp_f/np.sum(exp_f) #dim: 1 * C
      true_class = y[i]
      loss -= np.log(p[true_class])
      for j in range(C):
        dW[:,j] += (p[j] - (j==true_class))*X[i]
      
  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.           #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
    
  dW /= N
  dW += reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  C = W.shape[1]

  
  p = X.dot(W)
  p -= np.max(p, axis=1, keepdims=True)
  exp_p = np.exp(p)
  p = exp_p/np.sum(exp_p, axis=1, keepdims=True)
  loss = np.mean(p[np.arange(N),y])
  loss += 0.5*reg*np.sum(W*W)
  
  z = np.zeros_like(p)
  z[np.arange(N),y] = 1
  dW = X.T.dot(z)
  dW /= N
  dW += reg*W
  #############################################################################
  # Compute the softmax loss and its gradient using no explicit loops.        #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  return loss, dW

