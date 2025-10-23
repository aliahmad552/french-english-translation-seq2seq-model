import tensorflow as tf
import numpy as np
import re

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    sentence = '<start> ' + sentence + ' <end>'
    return sentence

def translate_sentence(sentence, encoder, decoder, fr_tokenizer, eng_tokenizer, max_length_inp=71, max_length_targ=71):
    sentence = preprocess_sentence(sentence)
    inputs = [fr_tokenizer.word_index.get(w, fr_tokenizer.word_index.get('<unk>', 1)) for w in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = encoder.initialize_hidden_state()
    enc_out, enc_h, enc_c = encoder(inputs, hidden)
    dec_hidden = (enc_h, enc_c)
    dec_input = tf.expand_dims([eng_tokenizer.word_index['<start>']], 0)

    for _ in range(max_length_targ):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = eng_tokenizer.index_word.get(predicted_id, '')
        if predicted_word == '<end>':
            break
        result += predicted_word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)
    return result.strip(), sentence, None 