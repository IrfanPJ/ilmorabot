from main import ChatBot
import streamlit as st

st.title('ILMORA Assistant Bot')

# --- Session State Initialization ---
# This ensures each user gets their own chatbot instance and message history.
# The chatbot is created only once per session.
if "bot" not in st.session_state:
    try:
        # Create a new chatbot instance for this specific session
        st.session_state.bot = ChatBot()
    except Exception as e:
        # Handle potential errors during initialization (e.g., missing API keys)
        st.error(f"Failed to initialize the chatbot. Error: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hello! I'm the ILMORA AI Assistant. How can I help you learn about our AI solutions and services?"
    }]

# --- Display Chat History ---
# Display all the messages stored in the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Handle New User Input ---
if user_input := st.chat_input("Ask your question here..."):
    # Append user's message to the history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use the session-specific chatbot to get a response
            response = st.session_state.bot.ask(user_input)
            st.write(response)
    
    # Append the assistant's response to the history
    st.session_state.messages.append({"role": "assistant", "content": response})