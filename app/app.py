import streamlit as st
import torch
import torch.nn as nn
import math
import pickle

# Load the saved vocabulary
# with open('models/vocab.pkl', 'rb') as f:
#     vocab = pickle.load(f)
# # Load training data and model parameters
Data = pickle.load(open('../models/vocab.pkl', 'rb'))
vocab_size = Data['vocab_size']
emb_dim = Data['emb_dim']
hid_dim = Data['hid_dim']
num_layers = Data['num_layers']
dropout_rate = Data['dropout_rate']
tokenizer = Data['tokenizer']
vocab = Data['vocab']

# Define the LSTM model (same as in utils.py)
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim    = hid_dim
        self.emb_dim    = emb_dim
        
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hid_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim, self.hid_dim).uniform_(-init_range_other, init_range_other) #We
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim, self.hid_dim).uniform_(-init_range_other, init_range_other) #Wh
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
        
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach() #not to be used for gradient computation
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

# Load the trained model
def load_model(model_path, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate, device):
    model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            # Print the top predictions
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5)
            print("Top predictions:", [(vocab.get_itos()[idx], prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])])
            
            # Sample the next token
            prediction = torch.multinomial(probs, num_samples=1).item()
            
            # Avoid generating <unk> tokens
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if <eos> is generated
            if prediction == vocab['<eos>']:
                break
            
            # Append the predicted token to the sequence
            indices.append(prediction)

    # Convert indices back to tokens
    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return ' '.join(tokens)

# Streamlit app
def main():
    st.title("LSTM Language Model Text Generator")
    
    # Sidebar for user inputs
    st.sidebar.header("Settings")
    max_seq_len = st.sidebar.slider("Max Sequence Length", 10, 100, 50)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0)
    seed = st.sidebar.number_input("Random Seed", value=42)

    # Main input
    prompt = st.text_input("Enter your prompt:", "love is")

    # Load model and vocabulary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('../models/best-val-lstm_lm.pt', vocab_size, emb_dim, hid_dim, num_layers, dropout_rate, device=device)

    # Tokenizer function (simple whitespace tokenizer)
    def tokenizer(text):
        return text.split()

    # Generate text when the user clicks the button
    if st.button("Generate Text"):
        with st.spinner("Generating text..."):
            generated_text = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed)
        st.success("Generated Text:")
        st.write(generated_text)

if __name__ == "__main__":
    main()