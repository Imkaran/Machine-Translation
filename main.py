from DataPreprocessing import encoder_data_preporcessing, decoder_data_preporcessing
from save_input import save_training_data, load_pickle_file
from Machine_Translation_Architecture import sequence2sequence_model
import pandas as pd

data = pd.read_table('dataset/mar.txt', names=['english', 'marathi'])
data = data.iloc[10000:20000]

encoder_input_data, num_english_token = encoder_data_preporcessing(data.english)
decoder_input_data, decoded_target_data, num_marathi_token = decoder_data_preporcessing(data.marathi)

input_training_data = {
    'encoder_input_data': encoder_input_data,
    'num_english_token': num_english_token,
    'decoder_input_data': decoder_input_data,
    'num_marathi_token': num_marathi_token,
    'decoded_target_data': decoded_target_data
}

try:
    save_training_data(input_training_data)
    encoder_input_data, num_english_token, decoder_input_data, num_marathi_token, decoded_target_data = load_pickle_file()
except:
    print("File Size Exceeded !!!")

model = sequence2sequence_model(num_english_token, num_marathi_token)
model.summary()
model.fit(x=[encoder_input_data, decoder_input_data], y= decoded_target_data, epochs=5000, batch_size=128)
model.save('machine_translation.h5')
