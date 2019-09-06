import pickle

def save_training_data(input_training_data):

    encoder_input_data = input_training_data['encoder_input_data']
    num_english_token = input_training_data['num_english_token']
    decoder_input_data = input_training_data['decoder_input_data']
    num_marathi_token = input_training_data['num_marathi_token']
    decoded_target_data = input_training_data['decoded_target_data']

    pickle.dump((encoder_input_data, num_english_token, decoder_input_data, num_marathi_token, decoded_target_data),open('Training_Data.p', 'wb'))

def load_pickle_file():

    encoder_input_data, num_english_token, decoder_input_data, num_marathi_token, decoded_target_data = pickle.load(open('Training_Data.p', 'rb'))

    return encoder_input_data, num_english_token, decoder_input_data, num_marathi_token, decoded_target_data