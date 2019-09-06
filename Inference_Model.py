import pandas as pd
from keras import preprocessing
from keras.models import Model
from keras.layers import Dense, LSTM, Input,Embedding
from keras.activations import softmax
from keras.optimizers import adam
import numpy as np
import pickle

class Inference_Machine_Translation_Model():

    def __init__(self):
        # self.read_and_preprocess_data()
        # self.save_input_inference_data()
        self.load_pickle_file()
        self.training_model_arch()
        self.run()

    def read_and_preprocess_data(self):
        data = pd.read_table('dataset/mar.txt', names=['english', 'marathi'])
        self.encoder_data_preporcessing(data.english)
        self.decoder_data_preporcessing(data.marathi)

    def encoder_data_preporcessing(self, english_data):
        tokenizer = preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(english_data)
        tokenized_english_lines = tokenizer.texts_to_sequences(english_data)

        sentence_length_list = [len(sentence) for sentence in tokenized_english_lines]
        self.input_max_length_sentence = max(sentence_length_list)

        encoder_input_data = preprocessing.sequence.pad_sequences(tokenized_english_lines, maxlen=self.input_max_length_sentence,
                                                                  padding='post')

        self.eng_word_dict = tokenizer.word_index
        self.num_english_token = len(self.eng_word_dict) + 1


    def decoder_data_preporcessing(self, marathi_data):
        marathi_lines = list()
        for marathi_line in marathi_data:
            marathi_lines.append('<Start> ' + marathi_line + ' <End>')

        tokenizer = preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(marathi_lines)
        tokenized_marathi_lines = tokenizer.texts_to_sequences(marathi_lines)

        # sentence_length_list = [len(sentence) for sentence in tokenized_marathi_lines]
        sentence_length_list = list()
        for token_seq in tokenized_marathi_lines:
            sentence_length_list.append(len(token_seq))
        self.output_max_length_sentence = max(sentence_length_list)

        decoder_input_data = preprocessing.sequence.pad_sequences(tokenized_marathi_lines, maxlen=self.output_max_length_sentence,
                                                                  padding='post')

        self.mar_word_dict = tokenizer.word_index
        self.num_marathi_token = len(self.mar_word_dict) + 1


    def save_input_inference_data(self):
        inference_input_data = {
            'encoder_vocab_count': self.num_english_token,
            'input_maximum_length': self.input_max_length_sentence,
            'encoder_word_to_index': self.eng_word_dict,
            'decoder_vocab_count': self.num_marathi_token,
            'output_maximum_length': self.output_max_length_sentence,
            'decoder_word_to_index': self.mar_word_dict
        }

        pickle.dump(inference_input_data, open('./pickle_data/Inference_Data.p', 'wb'))

    def load_pickle_file(self):
       self.inference_input_data = pickle.load(open('./pickle_data/Inference_Data.p', 'rb'))

    def training_model_arch(self):
        self.encoder_input = Input(shape=(None,))
        self.encoder_embedding = Embedding(self.inference_input_data['encoder_vocab_count'], 256, mask_zero=True)(self.encoder_input)
        self.output, self.h_state, self.c_state = LSTM(128, return_state=True)(self.encoder_embedding)
        self.encoder_input_state = [self.h_state, self.c_state]

        self.decoder_input = Input(shape=(None,))
        self.decoder_embedding = Embedding(self.inference_input_data['decoder_vocab_count'], 256, mask_zero=True)(self.decoder_input)
        self.decoder_lstm = LSTM(128, return_state=True, return_sequences=True)
        self.decoder_output, _, _ = self.decoder_lstm(self.decoder_embedding, initial_state=self.encoder_input_state)
        self.decoder_dense = Dense(self.inference_input_data['decoder_vocab_count'], activation=softmax)
        self.output = self.decoder_dense(self.decoder_output)

        self.model = Model(inputs=[self.encoder_input, self.decoder_input], outputs=self.output)
        self.model.load_weights('machine_translation_with_dropout.h5')
        # adam_optimizer = adam(lr=0.01, beta_1=0.9, beta_2=0.99)
        # self.model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])


    def inference_model_architetcure(self):
        encoder_model = Model(self.encoder_input, self.encoder_input_state)

        decoder_state_h_input = Input(shape=(128,))
        decoder_state_c_input = Input(shape=(128,))
        decoder_initial_state = [decoder_state_h_input, decoder_state_c_input]

        decoder_output, state_h, state_c = self.decoder_lstm(self.decoder_embedding,
                                                             initial_state=decoder_initial_state)
        decoder_state = [state_h, state_c]
        decoder_output = self.decoder_dense(decoder_output)
        decoder_model = Model([self.decoder_input] + decoder_initial_state,
                              [decoder_output] + decoder_state)

        return encoder_model, decoder_model


    def string_to_token(self,string):
        words = string.lower().split()
        tokens = list()
        for word in words:
            tokens.append(self.inference_input_data['encoder_word_to_index'][word])

        return preprocessing.sequence.pad_sequences([tokens], maxlen=self.inference_input_data['input_maximum_length'], padding='post')

    def run(self):
        encoder_model, decoder_model = self.inference_model_architetcure()

        user_message = input('Enter English Sentence:')

        state_values = encoder_model.predict(self.string_to_token(user_message))
        target_input = np.zeros((1,1))
        target_input[0, 0] = self.inference_input_data['decoder_word_to_index']['start']
        end_sequence = False
        translated_message = ''
        while not end_sequence:
            decoder_output, h, c = decoder_model.predict([target_input] + state_values)
            sampled_word_index = np.argmax(decoder_output[0, -1, :])
            sampled_word = None
            for word, index in self.inference_input_data['decoder_word_to_index'].items():
                if sampled_word_index == index:
                    if word != 'end':
                        translated_message += word + ' '
                    sampled_word = word

            if sampled_word == 'end' or len(translated_message.split()) > self.inference_input_data['output_maximum_length']:
                end_sequence = True

            target_input = np.zeros((1, 1))
            target_input[0][0] = sampled_word_index
            state_values = [h, c]

        print('Marathi Translation: '+translated_message)

if __name__=="__main__":
    app = Inference_Machine_Translation_Model()