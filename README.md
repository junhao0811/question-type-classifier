# Question Type Classifier

Classifies questions by the type of answer they expect (e.g. *"Where is the Eiffel Tower?"* → **location**), comparing three generations of NLP models on the same task.

## Notebooks

| Notebook | Approach |
|---|---|
| `1_word_embeddings_textcnn.ipynb` | Turns words into vectors with **Word2Vec / GloVe** embeddings, then classifies with a **Text CNN** (1-D convolutions over word sequences) |
| `2_rnn_lstm_gru.ipynb` | Reads questions word by word with recurrent networks, comparing **vanilla RNN, LSTM and GRU** cells |
| `3_transformers_bert.ipynb` | Uses **Transformer** models with attention, including pre-trained **BERT** (`bert-base-uncased`) via Hugging Face |


## The idea in one line
- **Text CNN** skims for key phrases
- **RNNs** read left to right, carrying memory
- **Transformers** look at every word at once and learn which ones matter

## Tech
PyTorch · Hugging Face Transformers & Datasets · Gensim · scikit-learn · pandas

## Run it
Open the notebooks in **Google Colab** (GPU recommended). Data and pre-trained embeddings download automatically.
