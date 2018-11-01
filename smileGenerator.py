from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import numpy as np
import json
import h5py
from pkg_resources import resource_filename
from model import smile_lstm_model
from smiles_util import *
from data import *
import csv
import re

class smileGeneratorLSTM():
    config = {
        'n_lstm_stacks': 2,
        'n_lstm_cells':256,
        'n_tokens': 45,
        'length_x':100,
        'max_length': 100,
        'dim_embedding': 200,
        'dropout':0.2,
        'max_training_data': 100000
    }

    default_config = config.copy()

    def __init__(self, weights_path = None, checkpoint_path = None):
        #if weights_path is None:
        #    weights_path = resource_filename(__name__, 'model_lstm_smile_weights.h5')

        self.model = smile_lstm_model(cfg = self.config, weights_path = weights_path)

        self.tokens, self.token2idx, self.idx2token, n_tokens = tokenize([])

        self.checkpoint_path = checkpoint_path

    def train(self, smile_list, batch_size = 100, epochs = 10, validation_split = 0.1):

        length_x = self.config['length_x']

        smile_vec, tokens, token2idx, idx2token, smile_not_processed = conv_smile_to_vec(smile_list,max_len = length_x)
        # Update the tokens of generator
        self.tokens = tokens
        self.token2idx = token2idx
        self.idx2token = idx2token
        # Update n_tokens in config
        self.config.update({'n_tokens': len(tokens)})

        n_smiles = len(smile_vec)
        n_tokens = self.config['n_tokens']
        # Splits the tokenized smile vector by given max length and stores next character as target
        X,Y = get_data(smile_vec, n_smiles = n_smiles, Tx = length_x, stride = 10, n_training = self.config['max_training_data'])
        # Converts to one-hot vector
        x,y = vectorization(X, Y, n_tokens, Tx = length_x)
        model_t = self.model
        if self.checkpoint_path != None:
            earlystopper = EarlyStopping(patience = 10, verbose = 1)
            checkpointer = ModelCheckpoint(self.checkpoint_path, verbose=1, save_best_only=True)
            model_t.fit(np.asarray(X), y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split = validation_split,
                        callbacks=[earlystopper, checkpointer])
        else:
            # Fit the model
            model_t.fit(np.asarray(X), y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split = validation_split)

    # Generate SMILES
    def generate_smiles(self,temperature = 1.0,embeddings = True):

        max_length = self.config['max_length']
        length_x = self.config['length_x']
        n_tokens = self.config['n_tokens']
        token2idx = self.token2idx
        idx2token = self.idx2token

        x = np.zeros((1,length_x,n_tokens))
        X = np.zeros((1,length_x))
        smile_vec = []
        counter = 0

        # Start a new smile
        start_smile_idx = token2idx['<']
        end_smile_idx = token2idx['>']

        # Initial vector is of the form '000...000<'
        x[0,length_x-1,start_smile_idx] = 1
        X[0,length_x-1] = start_smile_idx
        smile_vec.append(idx2token[start_smile_idx])

        idx = -1
        while(idx != end_smile_idx and counter!=max_length):

            if embeddings == True:
                y = self.model.predict(X,verbose = 0)
            else:
                y = self.model.predict(x,verbose = 0)
            idx = sample(y[0],n_tokens,temperature)
            out = idx2token[idx]
            smile_vec.append(out)
            counter +=1

            # Overwrite
            x[0,:length_x-1,:] = x[0,1:,:] # Shift everything by 1 place
            x[0,length_x-1,:] = 0 # Reset the previous idx
            x[0,length_x-1,idx] = 1 # Set the new idx

            X[0,:length_x-1] = X[0,1:] # Shift everything by 1 place
            X[0,length_x-1] = idx # Set the predicted idx at the final position

            #print(''.join(smile_vec))
            #print(X)

        smile = ''.join(smile_vec)
        return smile

    # Generate sample of SMILES
    def generate_smiles_list(self,n_smiles, return_valid = True):
        smile_list = []
        for _ in range(n_smiles):
            smile_list.append(self.generate_smiles(temperature = 1.0)[1:-1])

        # Return only unique smiles
        smile_list = list(set(smile_list))

        # Convert them to canonical smiles
        canonical_smile_list = canonize_smiles(smile_list)
        canonical_smile_list = [cs for cs in canonical_smile_list if cs != '']
        print('SMILES generated: {0}, SMILES valid: {1}'.format(len(smile_list),len(canonical_smile_list)))

        if return_valid:
            s = canonical_smile_list
        else:
            s = smile_list
        return s

    # Save model
    def save_model(self, weights_path='model_lstm_smile_weights.h5'):
        self.model.save_weights(weights_path)

    # Load model
    def load_model(self, weights_path):
        self.model = smile_lstm_model(cfg = self.config, weights_path = weights_path)

    # Reset
    def reset(self):
        self.config = self.default_config.copy()
        self.__init__()
