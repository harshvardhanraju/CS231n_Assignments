#from builtins import range
#from builtins import object
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

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
       
        self.params = {}
        self.reg = reg
        self.D = input_dim
        self.M = hidden_dim #   #filters #bias units,size of bias vector
        self.C = num_classes

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        W1 = weight_scale * np.random.rand(self.D,self.M)
        W2 = weight_scale * np.random.rand(self.M,self.C)
        b1 = np.zeros(self.M,)
        b2 = np.zeros(self.C,)
        
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['b1'] = b1
        self.params['b2'] = b2
        
        
      
       
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
        self.N = X.shape[0]
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        
        affine_1_out, affine_1_cache = affine_forward(X, W1 , b1)
        relu_1_out,relu_1_cache = relu_forward(affine_1_out)
        
        affine_2_out, affine_2_cache = affine_forward(relu_1_out, W2, b2)
        scores = affine_2_out
        
       
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
        loss, dx = softmax_loss(scores, y)
        loss += 0.5*self.reg*np.sum(W1*W1) + 0.5*self.reg*np.sum(W2*W2)
        
        #M*C   N*M                 N*C
        dW2 = affine_2_cache[0].T.dot(dx)
        #C,   N*C            N
        db2 = dx.T.dot(np.ones(self.N))
        
        X_reshaped = affine_1_cache[0].reshape(affine_1_cache[0].shape[0],np.prod(affine_1_cache[0].shape[1:]))
        #D*M         N*D             N*C      M*C 
        #dW1 = np.dot( X_reshaped.T.dot(dx) , affine_2_cache[1].T)#*(relu_1_cache > 0) 
        dW1 = np.dot(X_reshaped.T , ( dx.dot(affine_2_cache[1].T) ) *(relu_1_cache > 0) ) 
        #M             M*C                  N*C          N,
        db1 = np.dot( affine_2_cache[1].dot(dx.T),np.ones(self.N))
         
        #regularization added
        dW1 += self.reg*W1
        dW2 += self.reg*W2
        
        
        grads['W1']=dW1
        grads['W2']=dW2
        grads['b1']=db1
        grads['b2']=db2
        
        
        
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
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
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
       
        self.D = input_dim
        self.M = hidden_dims #  list of no of neurons in each unit 
        self.C = num_classes
        
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        W_dim = [self.D] + self.M + [self.C]       
        for i in range(self.num_layers):
            n = str(i + 1)
            #                                       mean   variance     dims
            self.params['W' + n] = np.random.normal(0, weight_scale, (W_dim[i],W_dim[i + 1]))
            self.params['b' + n] = np.zeros(W_dim[i + 1])
        
        
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
       
        if self.use_batchnorm:
            print('Using batchnorm')
            self.bn_params = {'bn_param' + str(i + 1): {'mode': 'train',
                                                        'running_mean': np.zeros(W_dim[i + 1]),
                                                        'running_var': np.zeros(W_dim[i + 1])}
                                                         for i in xrange(self.num_layers - 1)}
            self.gammas = {'gamma' + str(i + 1):
                      np.ones(W_dim[i + 1]) for i in range(self.num_layers - 1)}
            self.betas = {'beta' + str(i + 1): np.zeros(W_dim[i + 1])
                     for i in range(self.num_layers - 1)}

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
        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param['mode'] = mode

        scores = None
        self.N = X.shape[0]
         
        cache = {}
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
        #{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
        out = X
        for i in xrange(self.num_layers - 1):
            n = str(i+1)
            W, b = self.params['W' + n], self.params['b' + n]
            #out, cache['L' + n] = affine_relu_forward(out, W, b)
            
            #affine
            #N,M        (x,w,d)                         N,D  D,M M
            out, cache['A' + n] = affine_forward(out, W, b)
            #batchnorm
            if self.use_batchnorm:
                gamma = self.gammas['gamma' + str(i+1)]
                beta = self.betas['beta' + str(i+1)]
                # N,M                                   N,M    M,     M,  
                out, cache['B' + n] = batchnorm_forward(out, gamma, beta, self.bn_params['bn_param' + str(i+1)])
            #relu
            #N,M                                     N,M
            out, cache['R' + n] = relu_forward(out)
            #dropout
            if self.use_dropout:
                #N,M                                          N,M
                out, cache['D' + n] = dropout_forward(out, self.dropout_param)
        #for last affine layer    
        W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
        out, cache['A' + str(self.num_layers)] = affine_forward(out, W, b)
        scores = out
            
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #temp = {} #used to store the temp calculations inside backprop,using just 1 element in dict and overwriting on it
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
        loss, dsig = softmax_loss(scores, y)
        
        for i in range(self.num_layers):
            #regularisation for all W terms added to loss
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i+1)] * self.params['W' + str(i+1)])  
        
        #backprop through                                                                              
        dx, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)]  = affine_backward(dsig, cache['A'                                                                                                               + str(self.num_layers)])
        grads['W' + str(self.num_layers)] += self.reg * cache['A' + str(self.num_layers)][1]
        
        for i in range(self.num_layers - 1):
            #dx, grads['W' + str(self.num_layers - i -1)], grads['b' + str(self.num_layers - i -1)] = affine_relu_backward(dx,                                                                                       cache['L' + str(self.num_layers - i -1)])
                                                                                                #[affine_cache][W]
            if self.use_dropout:
                dx = dropout_backward(dx, cache['D' + str(self.num_layers - i -1)])
            
            dx = relu_backward(dx, cache['R' + str(self.num_layers - i -1)]) 
            if self.use_batchnorm:
                dx, dgamma, dbeta = batchnorm_backward(dx, cache['B' + str(self.num_layers - i -1)]) 
            
            dx, grads['W' + str(self.num_layers - i -1)], db = affine_backward(dx, cache['A' + str(self.num_layers - i -1)])
                                      
            
            grads['W' + str(self.num_layers - i -1)] += self.reg * cache['A' + str(self.num_layers - i -1)][1] 
            grads['b' + str(self.num_layers - i -1)] = db
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
