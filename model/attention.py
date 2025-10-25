import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense
import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
  def __init__(self,units):
    super(BahdanauAttention,self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)
  def call(self,encoder_outputs,decoder_hidden):
    decoder_hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)
    score = self.V(tf.nn.tanh(self.W1(encoder_outputs)+self.W2(decoder_hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score,axis = 1)
    context_vector = attention_weights * encoder_outputs
    context_vector = tf.reduce_sum(context_vector,axis = 1)
    return context_vector, tf.squeeze(attention_weights,axis = -1)