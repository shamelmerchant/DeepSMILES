import numpy as np
import random

# Generate one hot encoding of each SMILE
def one_hot_encode(smile_vec,n_token):
    encoding = list()
    for value in smile_vec:
        vector = [0 for _ in range(n_token)]
        vector[int(value)] = 1
        encoding.append(vector)

    return np.array(encoding)

# Decode a one-hot encoded sequence
def one_hot_decode(encoded_seq):
    if len(encoded_seq.shape) == 1:
        decoded_seq = np.array([np.argmax(encoded_seq)])
    else:
        decoded_seq = np.array([np.argmax(vector) for vector in encoded_seq])
    return decoded_seq

#
def get_data(smiles_vec, n_smiles = 1, Tx = 30, stride = 10, n_training = 1E6):
    X = []
    Y = []

    for i in range(0,n_smiles):
        for j in range(0,Tx,stride):
            X_vec = np.zeros((Tx))
            X_vec[j+1:] = smiles_vec[i,:Tx-j-1]
            Y_vec = smiles_vec[i,Tx-j-1]
            if Y_vec != 0:
                X.append(X_vec)
                Y.append(Y_vec)
    if n_training < len(X):
        # We will down sample to n_training data points
        idx = random.sample(list(np.arange(0,len(X))),int(n_training))
        # # TODO: Find a better way
        X = list(np.asarray(X)[idx,:])
        Y = list(np.asarray(Y)[idx])

    return X,Y
#
def vectorization(X, Y, n_tokens, Tx = 30):
    m = len(X)

    x = np.zeros((m, Tx, n_tokens), dtype=np.bool)
    y = np.zeros((m, n_tokens), dtype=np.bool)

    for i, smile in enumerate(X):
        for t, val in enumerate(smile):
            x[i,t,int(val)] = 1
        y[i,int(Y[i])] = 1

    return x, y

# Sampling using a probability distribution instead of argmax
def sample(preds, n_tokens, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    prob = np.random.multinomial(1,preds,1)
    out = np.random.choice(range(n_tokens), p = prob.ravel())
    return out
