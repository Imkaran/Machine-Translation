# Machine-Translation
Machine Translation using Keras

### Model Architetcure:

![encoder_decoder_model](https://user-images.githubusercontent.com/31925932/64406964-c3cf5380-d0a0-11e9-8b1b-a8161ebf84c0.PNG)

1. Encoder :
  * Encoder will take in the input as english sentence and return a hidden state
 
2. Decoder :
  * Decoder takes the input as hidden state from encoder and try to predict marathi transalation of english sentence
  
### Dataset :
[Click here](http://www.manythings.org/anki/mar-eng.zip) to download the dataset.It consist of total 35832 sentences which are translated from english to marathi

### Train Model:
> python train.py

##### Note: If code throws a MemoryError, then one hot encode the output in batches

### Inference Model:
> python Inference_Model.py

##### Note: I have saved the data in pickle file and load it. you need to first save the data in pickle file,so uncomment the line #13 and #14

### Accuarcy after 30 epochs:
![accuracy after 30 epochs](https://user-images.githubusercontent.com/31925932/64409020-b6689800-d0a5-11e9-8d08-534c33aa81a2.PNG)

