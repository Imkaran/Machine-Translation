from keras.layers import LSTM, Dense, Embedding
from keras import Input
from keras.models import Model
from keras.optimizers import adam
from keras.activations import softmax

def sequence2sequence_model(num_english_token, num_marathi_token):

    encoder_input = Input(shape=(None,))
    embedding_output = Embedding(num_english_token, 256, mask_zero=True)(encoder_input)
    output, h_state, c_state = LSTM(128, return_state=True)(embedding_output)
    encoder_input_state = [h_state, c_state]

    decoder_input = Input(shape=(None,))
    embedding_output = Embedding(num_marathi_token, 256, mask_zero=True)(decoder_input)
    decoder_lstm = LSTM(128, return_state=True, return_sequences=True)
    decoder_output, _, _ = decoder_lstm(embedding_output, initial_state = encoder_input_state)
    decoder_dense = Dense(num_marathi_token, activation=softmax)
    output = decoder_dense(decoder_output)

    model = Model(inputs= [encoder_input, decoder_input], outputs=output)
    adam_optimizer = adam(lr = 0.01, beta_1=0.9, beta_2=0.99)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    return model



