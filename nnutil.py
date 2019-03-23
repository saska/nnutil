import numpy as np

class NN:
    """
    Arguments: 
    data: data
    labels: labels
    layers: List (of lists) of net layer sizes and activation functions, e.g.
        [[8,"relu"],
         [5,"relu"],
         [3,"relu"],
         [2, "sigmoid"]]
         
        Currently supported functions: "relu", "tanh", "sigmoid"
        Notes:
        - Need to pass data or array-like of similar shape on initialization for creation of first layer 
        - Currently only works with sigmoid activation in the last layer due to
          the cost function partial derivative
    learning_rate: learning rate
    
    Uses heuristic initialization similar to Xavier (initial weights multiplied by np.sqrt(2/layer_sizes[i-1]))
    Uses cross-entropy cost
    """
    def __init__(self,  
                 layers,
                 data,
                 labels,
                 learning_rate):
        self.layers = layers
        self.data = data
        self.labels = labels
        self.learning_rate = learning_rate
        self.params = self.init_params()
    
    def sigmoid(self, Z):
        #Also returns original to help with backprop
        return 1/(1+np.exp(-Z)), Z

    def d_sigmoid(self, dA, cache):
        s, _ = self.sigmoid(cache)
        dZ = dA * s * (1-s)
        assert (dZ.shape == cache.shape)
        return dZ
    
    def relu(self, Z):
        #Also returns original to help with backprop    
        return Z.clip(min=0), Z

    def d_relu(self, dA, cache):
        dZ = np.array(dA, copy=True)
        dZ[cache <= 0] = 0
        assert (dZ.shape == cache.shape)
        return dZ
        
    def tanh(self, Z):
        #Also returns original to help with backprop
        A, _ = (self.sigmoid(Z * 2) * 2) - 1
        return A, Z
    
    def d_tanh(self, dA, cache):
        t, _ = self.tanh(cache)
        dZ = dA * (1 - t**2)
        assert (dZ.shape == cache.shape)
        return dZ
    
    def init_params(self):
        layer_sizes = [item[0] for item in self.layers]
        layer_sizes.insert(0, self.data.shape[0])
        params = {}
        
        for l in range(1,len(layer_sizes)):
            params['W' + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * np.sqrt(2/self.data.shape[1])
            params['b' + str(l)] = np.zeros((layer_sizes[l], 1))
        return params

    def forward_linear_step(self, A, W, b):
        Z = np.dot(W, A) + b
        return Z, (A, W, b)

    def forward_activation_step(self, A_prev, W, b, function):
        
        Z, lin_cache = self.forward_linear_step(A_prev, W, b)

        assert (function in ["relu", "sigmoid", "tanh"])
        A, act_cache = getattr(self, function)(Z)
        
        return A, (lin_cache, act_cache)
        
    def model_forward(self, X):
        caches = []
        A = X
        funcs = [item[1] for item in self.layers]
        L = len(self.params) // 2
        assert (len(funcs) == L)
        for l in range(L):
            A_prev = A
            A, cache = self.forward_activation_step(A_prev, self.params['W' + str(l+1)], self.params['b' + str(l+1)], funcs[l])
            caches.append(cache)
        
        return A, caches
    
    def cross_entropy_cost(self, AL, Y):
        cost = -np.mean(Y*np.log(AL) + (1-Y)*np.log(1-AL))
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        return cost
    
    def backward_linear_step(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db
    
    def backward_activation_step(self, dA, cache, function):
        
        lin_cache, act_cache = cache
        
        assert (function in ["relu", "sigmoid", "tanh"])
        function = str("d_" + function)
        dZ = getattr(self, function)(dA, act_cache)
        dA_prev, dW, db = self.backward_linear_step(dZ, lin_cache)
        
        return dA_prev, dW, db
    
    def model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
    
        grads["dA" + str(L)] = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        funcs = [item[1] for item in self.layers]
        assert (len(funcs) == L)
        
        for l in reversed(range(L)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.backward_activation_step(grads["dA" + str(l+1)], current_cache, funcs[l])
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        
        return grads
        
    def gradient_descent_update(self, grads):
        L = len(self.params) // 2
        for l in range(L):
            self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - grads["dW" + str(l+1)] * self.learning_rate
            self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - grads["db" + str(l+1)] * self.learning_rate
        
    def train(self, iterations, verbose=False):
        costs = []
        for i in range(0, iterations):
            AL, caches = self.model_forward(self.data)
            cost = self.cross_entropy_cost(AL, self.labels)
            grads = self.model_backward(AL, self.labels, caches)
            self.gradient_descent_update(grads)
            if i % 100 == 0:
                if verbose:
                    print ("Cost after iteration %i: %f" % (i, cost), end='\r')
                costs.append(cost)
                
        return costs, grads

def minibatch_gen_from_pddf(data, target_label, batch_size, shuffle=True):
    """
    Args: 
        data: data as pandas df
        target_label: target label column name in df
        batch_size: batch size
        shuffle: whether to shuffle the data.

    Yields:
        Data in num_batches equal batches with the last one (possibly) shorter  
    """
    target = np.array(data.pop(target_label))
    data = np.array(data)
    
    if shuffle:
        perm = np.random.permutation(len(target))
        target, data = target[perm], data[perm]
        
    num_batches = int(np.ceil(len(target) / batch_size))
    
    for i in range(1,num_batches+1):
        yield data[(i-1)*batch_size:i*batch_size, :], \
              target[(i-1)*batch_size:i*batch_size]