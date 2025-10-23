from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
from model.translate import translate_sentence

app = FastAPI(title="Neural Machine Translation API")

# Load tokenizers
with open("fr_tokenizer.pkl", "rb") as f:
    fr_tokenizer = pickle.load(f)
with open("eng_tokenizer.pkl", "rb") as f:
    eng_tokenizer = pickle.load(f)

# Load encoder and decoder
from model.encoder import Encoder
from model.decoder import Decoder

embedding_dim = 256
units = 512
vocab_inp_size = len(fr_tokenizer.word_index) + 1
vocab_tar_size = len(eng_tokenizer.word_index) + 1
BATCH_SIZE = 1

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

encoder.load_weights("encoder.h5")
decoder.load_weights("decoder.h5")

class TranslationInput(BaseModel):
    text: str

@app.post("/translate/")
def translate_text(data: TranslationInput):
    result, sentence, _ = translate_sentence(
        data.text, encoder, decoder, fr_tokenizer, eng_tokenizer
    )
    return {
        "input_french": sentence,
        "predicted_english": result
    }

@app.get("/")
def home():
    return {"message": "Welcome to the Neural Machine Translation API (French → English)"}
