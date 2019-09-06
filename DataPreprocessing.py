from keras import preprocessing, layers, activations, models, utils
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def encoder_data_preporcessing(english_data):

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(english_data)
    tokenized_english_lines = tokenizer.texts_to_sequences(english_data)

    sentence_length_list = [len(sentence) for sentence in tokenized_english_lines]
    max_length_sentence = max(sentence_length_list)

    encoder_input_data = preprocessing.sequence.pad_sequences(tokenized_english_lines, maxlen=max_length_sentence, padding='post')

    eng_word_dict = tokenizer.word_index
    num_english_token = len(eng_word_dict) + 1

    return encoder_input_data, num_english_token


def decoder_data_preporcessing(marathi_data):
    marathi_lines = list()
    for marathi_line in marathi_data:
        marathi_lines.append('<Start> '+marathi_line+' <End>')

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(marathi_lines)
    tokenized_marathi_lines = tokenizer.texts_to_sequences(marathi_lines)

    # sentence_length_list = [len(sentence) for sentence in tokenized_marathi_lines]
    sentence_length_list = list()
    for token_seq in tokenized_marathi_lines:
        sentence_length_list.append(len(token_seq))
    max_length_sentence = max(sentence_length_list)

    decoder_input_data = preprocessing.sequence.pad_sequences(tokenized_marathi_lines, maxlen=max_length_sentence, padding='post')

    mar_word_dict = tokenizer.word_index
    num_marathi_token = len(mar_word_dict) + 1

    #Prepare Target Data
    decoded_target_data = list()
    for each_sequence in tokenized_marathi_lines:
        decoded_target_data.append(each_sequence[1:])

    padded_marathi_data = preprocessing.sequence.pad_sequences(decoded_target_data, maxlen=max_length_sentence,
                                                               padding='post')

    decoded_target_data = np.array(utils.to_categorical(padded_marathi_data, num_marathi_token))
    # decoded_target_data = (np.arange(num_marathi_token) == padded_marathi_data[..., None]).astype(float)

    return decoder_input_data, decoded_target_data, num_marathi_token


