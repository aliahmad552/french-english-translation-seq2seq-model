from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import os
from fastapi.staticfiles import StaticFiles
import pickle
from model.translate import translate_sentence
from model.encoder import Encoder
from model.decoder import Decoder

app = FastAPI(title="Neural Machine Translation API")

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ---- Load tokenizers ----
with open("artifacts/fr_tokenizer.pkl", "rb") as f:
    fr_tokenizer = pickle.load(f)
with open("artifacts/eng_tokenizer.pkl", "rb") as f:
    eng_tokenizer = pickle.load(f)

# ---- Model setup ----
embedding_dim = 256
units = 512
vocab_inp_size = len(fr_tokenizer.word_index) + 1
vocab_tar_size = len(eng_tokenizer.word_index) + 1
BATCH_SIZE = 1

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# Build model once
sample_input = tf.zeros((1, 69))
sample_hidden = [tf.zeros((1, 512)), tf.zeros((1, 512))]
enc_output, enc_h, enc_c = encoder(sample_input, sample_hidden)
dec_input = tf.zeros((1, 1))
dec_hidden = (enc_h, enc_c)
_ = decoder(dec_input, dec_hidden, enc_output)

encoder.load_weights("artifacts/encoder.weights.h5")
decoder.load_weights("artifacts/decoder.weights.h5")

# ---- Input Model ----
class TranslationInput(BaseModel):
    text: str

# ---- API ROUTES ----
@app.post("/translate/")
def translate_text(data: TranslationInput):
    result, sentence, _ = translate_sentence(
        data.text, encoder, decoder, fr_tokenizer, eng_tokenizer
    )
    return {
        "input_french": sentence,
        "predicted_english": result
    }

@app.get("/api")
def home():
    return {"message": "Neural Machine Translation API is running 🚀"}

# ✅ Mount frontend LAST (after all API routes)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
