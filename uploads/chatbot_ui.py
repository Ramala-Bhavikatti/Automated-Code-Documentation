import streamlit as st
import requests

server_url = "http://127.0.0.1:5000/chat"

st.title("Conversational Chatbot")
user_message = st.text_input("You:")

len_of_user_message = len(user_message)

if st.button("Send"):
    response = requests.post(server_url, json={"message": user_message})
    if response.status_code == 200:
        bot_message = response.json()["response"]
        st.write(f"Bot: {bot_message[len_of_user_message: ]}")
