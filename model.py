from keras.models import Model
from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from keras.layers import concatenate, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Import attention layer (Adapted from DeepEmoji)
from AttentionWeightedAverage import AttentionWeightedAverage

def smile_lstm_model(cfg, weights_path = None, summary = True):
    '''
    Builds stacked lstm model for SMILE generator and loads weights if given
    '''

    length_x = cfg['length_x']
    n_tokens = cfg['n_tokens']
    dim_embedding = cfg['dim_embedding']
    n_lstm_stacks = cfg['n_lstm_stacks']
    n_lstm_cells = cfg['n_lstm_cells']
    dropout = cfg['dropout']

    lstm_stack = []

    input = Input(shape = (length_x,))
    embedded = Embedding(n_tokens, dim_embedding, input_length = None) (input)
    if dropout > 0.0:
        embedded = Dropout(dropout)(embedded)
    for i in range(n_lstm_stacks):
        if i == 0:
            lstm_stack.append(Bidirectional(LSTM(n_lstm_cells, input_shape=(None,n_tokens), return_sequences = True))(embedded))
        else:
            lstm_stack.append(Bidirectional(LSTM(n_lstm_cells, input_shape=(None,n_tokens), return_sequences = True))(lstm_stack[i-1]))

    attention = AttentionWeightedAverage(name='attention')(lstm_stack[i])
    output = Dense(n_tokens,activation='softmax')(attention)

    model = Model(inputs = [input], outputs = [output])
    if weights_path is not None:
        model.load_weights(weights_path)

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    if summary:
        model.summary()

    return model
