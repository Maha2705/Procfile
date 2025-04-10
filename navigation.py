import streamlit as st
import base64
import os
import sqlite3
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")
st.markdown(
    """
    <style>
    .nav-links {
        text-align: right;
        padding: 10px;
    }
    .nav-links a {
        text-decoration: none;
        padding: 8px 15px;
        font-weight: bold;
        color: #d35400;
        border: 1px solid #d35400;
        border-radius: 5px;
        margin-left: 10px;
    }
    .nav-links a:hover {
        background-color: #d35400;
        color: white;
    }
    </style>
    <div class='nav-links'>
        <a href='?p=home'>🏠 Home</a>
        <a href='?p=reg'>📝 Register</a>
        <a href='?p=log'>🔐 Login</a>
    </div>
    """,
    unsafe_allow_html=True
)
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_string.decode()});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

def navigation():
    return st.query_params.get('p', ['home'])[0]

page = navigation()

# ✅ Debug print to see what page is loaded
st.write(f"📌 You are on page: `{page}`")
if page == "home":
    add_bg_from_local("1.jpg")
    st.markdown("<h1 style='text-align:center; color:#d35400;'>Green Options for Milk Packaging using Intelligent Packaging</h1>", unsafe_allow_html=True)
    st.markdown("...")
    
elif page == "reg":
    add_bg_from_local("reg.avif")
    st.markdown("## Register Here")
    # Registration form code here...

elif page == "log":
    add_bg_from_local("login.jpg")
    st.markdown("## Login Here")
    name = st.text_input("Username")
    password = st.text_input("Password", type="password")
    conn = sqlite3.connect("dbs.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        password TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        phone TEXT NOT NULL);""")
    if st.button("Login"):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))
        user = cur.fetchone()
        if user:
            st.success(f"Welcome back, {user[1]}!")
            try:
                with open("Prediction.py") as f:
                    exec(f.read())
            except FileNotFoundError:
                st.error("Prediction.py not found.")
        else:
            st.error("Invalid username or password.")
    conn.close()

else:
    st.warning("⚠️ Page not found. Please use the menu above.")
