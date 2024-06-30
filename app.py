import streamlit as st
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_path = "./fine_tuned_model"  # Path to your model
    tokenizer = BlenderbotTokenizer.from_pretrained(model_path)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_response(query, max_length=150):
    inputs = tokenizer([query], return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=True, top_p=0.9, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("Nexforce Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
