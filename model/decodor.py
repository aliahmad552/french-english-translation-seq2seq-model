class Decoder(tf.keras.Model):
  def __init__(self,vocab_size,embedding_dim,dec_units,batch_size):
    super(Decoder,self).__init__()
    self.batch_size = batch_size
    self.enc_units = dec_units
    self.embedding = Embedding(vocab_size,embedding_dim)
    self.lstm = LSTM(self.enc_units,return_sequences = True,return_state = True,recurrent_initializer = 'glorot_uniform')
    self.fc = Dense(vocab_size)
    self.attention = BahdanauAttention(self.enc_units)
  def call(self,x,hidden,encoder_outputs):
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    context_vector,attention_weights = self.attention(encoder_outputs,hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector,1),x],axis = -1)
    output,state_h,state_c = self.lstm(x)
    output = tf.reshape(output,(-1,output.shape[2]))
    x = self.fc(output)
    return x,(state_h,state_c),attention_weights