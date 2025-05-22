import streamlit as st

st.set_page_config(page_title="My Fuzzy App", layout="wide")

st.title("Hello World")
st.write("This is a simple app")


x = st.text_input("Enter your name", key="name")
st.write("Your name is:", x)
