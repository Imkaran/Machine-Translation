# Machine-Translation
Machine Translation using Keras

### Model Architetcure:

![encoder_decoder_model](https://user-images.githubusercontent.com/31925932/64406964-c3cf5380-d0a0-11e9-8b1b-a8161ebf84c0.PNG)

1. Encoder :
  *Encoder will take in the input as english sentence and return a hidden state
 
2. Decoder :
  *Decoder takes the input as hidden state from encoder and try to predict marathi transalation of english sentence
  
  
### Train Model:
> python main.py

##### Note: If code throws a MemoryError, then one hot encode the output in batches

### Inference Model:
> python Inference_Model.py

##### Note: 
I have saved the data in pickle file and load it. you need to first save the data in pickle file,so uncomment the line #13 and #14
