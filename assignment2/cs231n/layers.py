#from builtins import range
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
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_row = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    #N,M 
    out = x_row.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
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

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    
    
    dx = (dout.dot(w.T)).reshape(x.shape)
    #print(  dout.T.dot(x).shape)
    dw = np.dot(x.reshape(x.shape[0], np.prod(x.shape[1:])).T,dout)

    db = dout.T.dot(np.ones(dout.shape[0]))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
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
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    relu_mask = x > 0
    dx = dout * relu_mask
    ###########################################################################
    #                             END OF YOUR CODE                            #
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
    eps = bn_param.get('eps', 1e-7)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
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
        #######################################################################
        '''
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        #sample_var = (1 / N) * np.sum((x - sample_mean)**2, axis = 0)
        norm_x = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = norm_x * gamma + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var 
        
        cache = x, sample_mean, sample_var, gamma, eps #backpass through gamma/sample_var  
        '''
        
        #(D,)
        sample_mean = 1 / float(N) * np.sum(x, axis=0)
        #(N,D)
        num = x - sample_mean
        num_sqr = num**2
        #(D,)
        sample_var = diff_mean = 1 / float(N) * np.sum(num_sqr, axis=0)
        den_sqrt = (sample_var + eps)**0.5
        #dsample_var = 0.5 * (sample_var + eps)**-0.5
        den = 1.0 / den_sqrt
        #dden_sqrt = -1 / (den_sqrt**2)
        
        #N,D
        norm_x = num * den
        #dnum, dden = den, num
        out = norm_x * gamma + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        cache = gamma, sample_mean, sample_var, eps, num, den, den_sqrt, num_sqr
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = (x - running_mean) / np.sqrt(running_var + eps)
        out = out * gamma + beta 
        
        #######################################################################
        #                          END OF YOUR CODE                           #
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
  
    dx = np.zeros_like(dout)
    N, D = dout.shape
    gamma, sample_mean, sample_var, eps, num, deno, den_sqrt, num_sqr = cache  
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    #D,     N,D            N
    #dbeta = dout.T.dot(np.ones(N,))
    dbeta = np.sum(dout, axis=0)
    #D,             N,D     N,D    D,
    dgamma = np.sum(dout * (num * deno), axis=0) #here deno is 1/(var+eps)**0.5
    
    #N,D     N,D    D,
    dback = dout * gamma 
    #N,D     D     N,D
    dnum = deno * dback
    dx = dnum
    #D,                 N,D
    dmean = -1. * np.sum(dnum, axis=0)
    
   
    #D,             N,D    N,D
    ddeno = np.sum(dback * num, axis=0)
    #D,                   D            D,
    dden_sqrt = (-1. / (den_sqrt**2)) * ddeno
    #D,                      D,                          D,
    dsample_var = (0.5 * ((sample_var + eps)**(-0.5))) * dden_sqrt
    #N,D                                    D,
    dnum_sqr = (1. / N) * np.ones((N,D)) * dsample_var
    #N,D        N,D       N,D
    dnume = 2. * num * dnum_sqr
    #N,D       N,D
    dx += dnume   #same as reshape(using np.ones) into N,D and addition
    #D,                    N,D
    dmean += -1.  * np.sum(dnume, axis=0) 
    #N,D              N,D              D,
    dx  += (1. / N) * np.ones((N,D)) * dmean
    
    '''
    #my u/v method
    du_x = 1 / N
    dvar_x = (1 / N) * 2 * (x - mean) * (1 - 1/N)
    ddeno_x = 0.5 * ((variance + eps)**-0.5) * dvar_x
    dnorm_x = ((1 - du_x) * ((variance + eps)**0.5) - ddeno_x * (x - mean)) / (variance + eps)  
    dx = dout * gamma * (dnorm_x)
    '''
    #my  graph method
    #df_var = 2 * N * (x - mean) * ( -0.5 * (variance + eps)**-1.5)
    #df_x = (variance + eps)**-0.5
    #df_u = N * (-1 * (variance + eps)**-0.5 - df_var)
    #dx = dout * gamma * (df_var + df_x + df_u)
    
    #dx = dout * gamma * (1 - (1/N)) * (1 - variance) / np.sqrt(variance + eps)
    
    
    #dx = ((N - 1) / N) * ((1 / (variance + eps)**0.5) - ((x - mean)**2 / (variance + eps)**1.5))   
    #dx = dx * gamma *dout
    '''
    dvar = dout * gamma * (x - mean)
    ddeno = -dvar * (1 / (variance + eps)**0.5) 
    droot = ddeno * 0.5 * ((variance + eps)**-0.5)
    dsum = droot * (1 / N) 
    dsqr = dsum * 2 * (x - mean)
    dnum = dout * gamma * (variance + eps)**-0.5
    dcombined = dnum + dsqr
    dx1 = dcombined
    dx2 = -dcombined * (1 / N)
    
    dx = dx1 + dx2
    '''
    
    #dden = (num) * dnorm_x * dden * dden_sqrt * (1 / N) * dnum_sqr 
    #dnum = ((variance + eps)**-0.5) * dnorm_x 
    #d_combined = dnum + dden
    #dx = d_combined * (1 - (1 / N)) 
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    dx = np.zeros_like(dout)
    N, D = dout.shape
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    gamma, sample_mean, sample_var, eps, num, deno, den_sqrt, num_sqr = cache
    
    dbeta = dout.T.dot(np.ones(N,))
    dgamma = np.sum(dout * (num * deno), axis=0)
    #N,D                      D,                                           D,                       N,D                        
    dx = dout * gamma * (1. - (1. / N)) * ((sample_var + eps)**0.5) * (1. - ((1. / N) * ((sample_var + eps)**-1) * ((num)**2)))
    #dx = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dout - np.sum(dout, axis=0)
                                                       #- (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=0))
    print(dx.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p 
        out= x * mask
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
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
    p, mode = dropout_param['p'], dropout_param['mode']
    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    cache = (x, w, b, conv_param) #original non zero padded values stored
    out = None
    S, P = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    x_new = ((W - WW + 2*P) / S) + 1 #W'
    y_new = ((H - HH + 2*P) / S) + 1 #H'
    v = np.zeros((N,F,y_new,x_new))
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    #zero padding
    x = np.lib.pad(x, ((0,0), (0,0), (P, P), (P, P)),'constant') # pad only across x and y direction,not along depth / #examples
    #print(x.shape)
    
    for n in xrange(N):
        for k in xrange(F): # calc the conv for F filters , w & b will be diff for diff F and will share parameters otherwise
            for j in xrange(y_new): 
                for i in xrange(x_new):
                    v[n,k,j,i] = np.sum(x[n, :, j * S : j * S + HH, i * S : i * S + WW] * w[k]) + b[k]
         
    out = v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    #cache = (x, w, b, conv_param) #zero padded x is stored here
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    
    ###########################################################################
    # TODO: Implement the convolutional backward pcol.                        #
    ##################################################col######################
    
    dw = np.zeros_like(w)
    
    S, P = conv_param['stride'], conv_param['pad']
    x = np.lib.pad(x, ((0,0), (0,0), (P, P), (P, P)),'constant') # pad only across x and y direction,not along depth 
    N, C, H, W = x.shape
    F, C_, HH, WW = w.shape
    N_, F_, Hd, Wd = dout.shape  # Hd = (W - WW + 2*P) / S) + 1 i.e zero padded size 
    dx = np.zeros_like(x)
    
    for f in xrange(F):
        for c in xrange(C): 
            for j in xrange(HH): 
                for i in xrange(WW):
                     #H-(HH-1)is coverage area of W1 ,sum across all N,as same W1 would be interacting with all the samples(N)
                     #1,1,1,1            N,1,Hd,Wd       N,1,H,W        
                    dw[f,c,j,i] = np.sum(dout[:, f,:,:] * x[:, c, j:j + (H - HH + 1):S, i:i + (W - WW + 1):S])
    
    db = np.sum(np.sum(np.sum(np.transpose(dout, (1,0,2,3)), axis=3), axis=2), axis=1)
    
    '''#works if conv operation preserves the shape and the o/p shape is not reduced after conv
    #convolution of dout with W flipped in both spatial directions
    #bad logic to zero pad, required for mathematical sense
    dout = np.lib.pad(dout, ((0,0), (0,0), (P, P), (P, P)),'constant')
    w_flip = np.flip(np.flip(w, axis=3), axis=2)
    
    for n in xrange(N):
        for c in xrange(C): 
            for j in xrange(H - 2*P): # H-2P is the original dim of x before zero pad 
                for i in range(W - 2*P):
                    print(i)
                    #sum across filters as same X11 interacts with all filters, so their contribution should add
                    #                         1, F, HH, WW                                      F, 1, HH, WW
                    dx[n,c,j,i] = np.sum(dout[n, :, j * S : j * S + HH:S, i * S : i * S + WW:S] * w_flip[:, c, ::S, ::S]) 
    
    '''
    #intuitively backprop through the notes visualisation
    for n in xrange(N):
        for c in xrange(C): 
            for j in xrange(Hd): # H-2P is the original dim of x before zero pad 
                for i in range(Wd):
                    #sum across filters as same X11 interacts with all filters, so their contribution should add
                    #look at a portion of x, update it with W * one value of dout which was affected from that region of x
                    #                                                                F,                           F,HH,WW
                    dx[n, c, j * S : j * S + HH, i * S : i * S + WW] += np.sum((dout[n, :, j, i].reshape(F,1,1) * w[:, c, :, :]), axis=0) 
                    
    
    dx = dx[:, :, P : H-P, P : W-P]    #remove contribution of zero padded elements            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    N, C, H, W = x.shape #C is the no of filters 
    pool_h, pool_w, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    h_new = ((H - pool_h) / S)+ 1
    w_new = ((W - pool_w) / S)+ 1
    out = np.zeros((N, C, h_new, w_new))
    for n in xrange(N):
        for k in xrange(C): # calc the conv for F filters , w & b will be diff for diff F and will share parameters otherwise
            for j in xrange(h_new): 
                for i in xrange(w_new):
                    out[n,k,j,i] = np.max(x[n, k, j * S : j * S + h_new, i * S : i * S + w_new])
                    #keep track of max indexes,create a mask and store in cache(mask*W*dout) and filter using max filter
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    dx = np.zeros_like(x)
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    N, C, H, W = x.shape #C is the no of filters 
    pool_h, pool_w, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    h_new = ((H - pool_h) / S)+ 1
    w_new = ((W - pool_w) / S)+ 1
    
    for n in xrange(N):
        for k in xrange(C): # calc the conv for F filters , w & b will be diff for diff F and will share parameters otherwise
            for j in xrange(h_new): 
                for i in xrange(w_new):
                    ROI = x[n, k, j * S : j * S + h_new, i * S : i * S + w_new]
                    #h_new,w_new
                    mask = (ROI == np.max(ROI)) #generates a mask 
                    dx[n, k, j * S : j * S + h_new, i * S : i * S + w_new] += dout[n, k, j, i] * mask
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
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

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
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

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
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
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    
    probs = np.exp(log_probs)
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
