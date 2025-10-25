# 🧠 Neural Machine Translation (NMT) – French to English Translator

This project is a **Neural Machine Translation (NMT)** system that translates French sentences into English using a **Sequence-to-Sequence (Seq2Seq)** architecture with **Attention Mechanism** and **Beam Search Decoding**. The model is built using **TensorFlow** and deployed through a **FastAPI backend** with a **HTML, CSS, and JavaScript frontend**.  

It simulates how real-world translation systems like **Google Translate** work — encoding a sentence in one language and decoding it into another using deep learning.

---

## 🚀 Project Overview

Language translation is one of the most fascinating tasks in Natural Language Processing (NLP). Traditional approaches like rule-based systems or statistical translation models fail to capture the contextual meaning of sentences. This project overcomes those limitations by implementing a **deep learning–based neural translation system**.

The model follows an **Encoder–Decoder** architecture where the Encoder reads and encodes the French sentence into a fixed-length vector representation, and the Decoder generates the English translation word by word.  
To further improve translation quality, the model uses an **Attention mechanism**, allowing it to focus on specific parts of the input sentence when predicting each word.  

Additionally, **Beam Search decoding** is used during inference to explore multiple translation possibilities simultaneously, resulting in smoother, more accurate translations compared to simple greedy decoding.

---

## 🧩 Model Architecture

The architecture of this project is divided into three key parts — **Encoder**, **Attention**, and **Decoder**:

1. **Encoder:**  
   The Encoder is a bidirectional LSTM network that reads the French sentence and converts it into context vectors. It captures both forward and backward dependencies, which helps understand the full meaning of the input sequence.

2. **Bahdanau Attention:**  
   The Attention mechanism computes a weighted sum of the encoder outputs for every decoder time step. This allows the model to focus on the most relevant parts of the input sentence while generating each English word.

3. **Decoder:**  
   The Decoder is also an LSTM network that uses the previous hidden states, the context vector from Attention, and the previously predicted words to generate the next word in the English translation.

---

## 🔍 Beam Search in Translation

During inference, a **Beam Search algorithm** is applied instead of greedy decoding.  
While greedy decoding only selects the single most probable word at each step, Beam Search keeps track of the top *k* most probable sequences (beams).  

For every new predicted word, the algorithm expands all current beams and selects the *k* best-scoring partial translations. This process continues until an `<end>` token is predicted or the maximum sequence length is reached.  
Beam Search significantly improves translation fluency and prevents the model from repeating phrases or getting stuck in loops.

---

## ⚙️ Backend – FastAPI

The backend is built using **FastAPI**, a modern and high-performance Python web framework. It exposes a `/translate/` POST endpoint where the frontend sends a French sentence, and the server responds with its English translation.

The backend workflow is as follows:
1. Accepts a French sentence as JSON input.
2. Preprocesses the sentence using the same tokenizer and sequence padding used during training.
3. Passes the input through the encoder and decoder models.
4. Generates the translated English sentence using beam search.
5. Returns the translated text as a JSON response.

The models (`encoder.weights.h5`, `decoder.weights.h5`) and tokenizers are loaded once when the server starts, ensuring fast translation during requests.

---

## 🎨 Frontend – HTML, CSS, and JavaScript

The frontend is a simple yet clean user interface designed to interact with the FastAPI backend. It contains a text box where the user can enter a French sentence, a “Translate” button to send the request, and an output section to display the translated English sentence.  

The JavaScript file (`script.js`) handles all frontend–backend communication using the Fetch API. When the user clicks the translate button:
- It sends a POST request to the FastAPI endpoint `/translate/` with the input sentence.
- It waits for the response and then displays the predicted translation in the output area.
- It also includes loading states and error handling to improve user experience.

---

## 🧠 Training and Model Files

The model was trained on a **French–English parallel corpus**, preprocessed with tokenization, padding, and word indexing using Keras `Tokenizer`. The training process involved:
- Converting sentences into numerical sequences.
- Padding them to uniform length.
- Using **Teacher Forcing** to guide the decoder during training.
- Optimizing with the **Adam optimizer** and **Sparse Categorical Crossentropy** loss.

**After training, the encoder and decoder weights were saved as:**

```bash
artifacts/encoder.weights.h5
artifacts/decoder.weights.h5
```
**Additionally, tokenizers were stored as pickle files:**
```bash
artifacts/fr_tokenizer.pkl
artifacts/eng_tokenizer.pkl
```


These files are later reloaded during inference in the FastAPI app.
## 🧰 Folder Structure
NMT-Project/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── model/
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   └── translate.py
│
├── artifacts/
│   ├── encoder.weights.h5
│   ├── decoder.weights.h5
│   ├── fr_tokenizer.pkl
│   └── eng_tokenizer.pkl
│
├── app.py
└── README.md

---
## 🧪 How to Run the Project

### Clone the repository:
```bash
git clone https://github.com/yourusername/nmt-translator.git
cd nmt-translator

```
### Create and activate a virtual environment:
```bash
venv\Scripts\activate  # for Windows
source venv/bin/activate  # for Mac/Linux
```
### Install dependencies:
```bash
pip install -r requirements.txt
```
### Run the FastAPI app:
```bash
uvicorn app:app --reload
```
### Open the frontend:

Once the server starts, open your browser and go to:
```bash
http://127.0.0.1:8000/
```

Enter any French sentence and click Translate to see the result

### 🔬 Future Improvements

- This project can be further enhanced by:

- Using Transformer-based models (like BERT or T5) for improved accuracy

- Integrating Subword tokenization (Byte Pair Encoding) to handle rare words

- Deploying the model as a Streamlit or Gradio web app for a smoother interface

- Using a larger dataset (like WMT’14 French–English) for higher translation quality

