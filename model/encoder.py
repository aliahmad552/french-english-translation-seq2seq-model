import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.enc_units = enc_units
    self.embedding = Embedding(vocab_size, embedding_dim)
    self.lstm = LSTM(self.enc_units,return_sequences = True,return_state = True,recurrent_initializer='glorot_uniform')
  def call(self,x,hidden):
    x = self.embedding(x)
    output,state_h,state_c = self.lstm(x,initial_state = hidden)
    return output,state_h,state_c
  def initial_hidden_state(self):
    return (tf.zeros((self.batch_size, self.enc_units)),
                tf.zeros((self.batch_size, self.enc_units)))
