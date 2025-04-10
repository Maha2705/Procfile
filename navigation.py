# ====================== IMPORT PACKAGES ==============
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import streamlit as st
import base64
import sqlite3
import re

warnings.filterwarnings("ignore")

# =================== NAVIGATION BAR ===================
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

# ================ BACKGROUND IMAGE FUNCTION ================
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

# ================ PAGE ROUTING ================
def navigation():
    return st.query_params.get('p', ['home'])[0]

page = navigation()

# =================== HOME PAGE ===================
if page == "home":
    add_bg_from_local('1.jpg')
    st.markdown("<h1 style='text-align:center; color:#d35400;'>Green Options for Milk Packaging using Intelligent Packaging</h1>", unsafe_allow_html=True)
    st.write("-------------------------------------------")
    st.markdown(
        "<h2 style='color:#000000; text-align: justify; font-family:Verdana; font-size:20px;'>"
        "The proposed system for green milk packaging uses machine learning to optimize and evaluate various materials. "
        "It integrates data from Bagasse quality, PLA production, and MAP datasets, performing extensive preprocessing and PCA. "
        "Models like Multi-layer Neural Networks, SVM, and Random Forest are trained and evaluated for accuracy, precision, recall, and F1 score. "
        "The system predicts packaging quality and shelf life, providing insights into the effectiveness of green packaging options."
        "</h2>",
        unsafe_allow_html=True
    )

# =================== REGISTER PAGE ===================
elif page == "reg":
    add_bg_from_local('reg.avif')
    st.markdown("<h1 style='text-align:center; color:#d35400;'>Register Here</h1>", unsafe_allow_html=True)

    def create_connection(db_file="dbs.db"):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)
        return conn

    def create_user(conn, user):
        sql = ''' INSERT INTO users(name, password, email, phone) VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()

    def user_exists(conn, email):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        return cur.fetchone() is not None

    def validate_email(email):
        return re.match(r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$', email)

    def validate_phone(phone):
        return re.match(r'^[6-9]\d{9}$', phone)

    conn = create_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    phone TEXT NOT NULL);''')

    name = st.text_input("Name")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    email = st.text_input("Email")
    phone = st.text_input("Phone")

    if st.button("Register"):
        if password == confirm_password:
            if not user_exists(conn, email):
                if validate_email(email) and validate_phone(phone):
                    create_user(conn, (name, password, email, phone))
                    st.success("User registered successfully!")
                else:
                    st.error("Invalid email or phone number!")
            else:
                st.error("User already exists!")
        else:
            st.error("Passwords do not match!")
    conn.close()

# =================== LOGIN PAGE ===================
elif page == "log":
    add_bg_from_local('login.jpg')
    st.markdown("<h1 style='text-align:center; color:#d35400;'>Login Here</h1>", unsafe_allow_html=True)

    def validate_user(conn, name, password):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))
        return cur.fetchone()

    conn = create_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    phone TEXT NOT NULL);''')

    st.write("Enter your credentials:")
    name = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = validate_user(conn, name, password)
        if user:
            st.success(f"Welcome back, {user[1]}! Login successful!")

            # ✅ Directly run Prediction.py content
            try:
                with open("Prediction.py") as f:
                    exec(f.read())
            except FileNotFoundError:
                st.error("Prediction.py file not found.")
        else:
            st.error("Invalid username or password.")
    conn.close()

# =================== INVALID PAGE ===================
else:
    st.error("❌ Page not found.")
