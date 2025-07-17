from ilmora import ChatBot
import streamlit as st

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return ChatBot()

bot = get_chatbot()

st.title('ILMORA Assistant Bot')

# Function for generating LLM response
def generate_response(input):
    return bot.ask(input)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ILMORA AI Assistant. How can I help you learn about ILMORA's AI solutions and services?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(input) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)