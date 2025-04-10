
# ====================== IMPORT PACKAGES ==============
   
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 
   
import streamlit as st
import base64
     
import sqlite3

# ================ Background image ===



def navigation():
    # Default to 'home' if no query param is provided
    path = st.query_params.get('p', ['home'])[0]
    return path




if navigation() == "home":
    
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('1.Jpg')

    st.markdown(f'<h1 style="color:#d35400 ;text-align: center;font-size:34px;font-family:verdana;">{"Green Options for milk packaging using intelligent packaging"}</h1>', unsafe_allow_html=True)
    st.write("-------------------------------------------")
    print()
    print()

    print()

    st.text("                 ")
    st.text("                 ")
    a = "  * The proposed system for green milk packaging uses machine learning to optimize and evaluate various materials. It integrates data from Bagasse quality, PLA production, and MAP datasets, performing extensive preprocessing and feature extraction via PCA. Models like Multi-layer Neural Networks, SVM, and Random Forest are trained and evaluated for accuracy, precision, recall, and F1 score. The system predicts packaging quality and shelf life, providing insights into the effectiveness of green packaging options. * "
    
    st.markdown(f'<h1 style="color:#000000;text-align: justify;font-size:30px;font-family:Caveat, sans-serif;">{a}</h1>', unsafe_allow_html=True)

    st.text("                 ")
    st.text("                 ")
    
    st.text("                 ")
    st.text("                 ")
    



elif navigation()=='reg':
    
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('reg.avif')

   
    st.markdown(f'<h1 style="color:#d35400 ;text-align: center;font-size:34px;font-family:verdana;">{"Green Options for milk packaging using intelligent packaging"}</h1>', unsafe_allow_html=True)
    st.write("-------------------------------------------")

    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:25px;font-family:Algerian;">{"Register Here !!!"}</h1>', unsafe_allow_html=True)
    
    import streamlit as st
    import sqlite3
    import re
    
    # Function to create a database connection
    def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)
        return conn
    
    # Function to create a new user
    def create_user(conn, user):
        sql = ''' INSERT INTO users(name, password, email, phone)
                  VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        return cur.lastrowid
    
    # Function to check if a user already exists
    def user_exists(conn, email):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        if cur.fetchone():
            return True
        return False
    
    # Function to validate email
    def validate_email(email):
        pattern = r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
        return re.match(pattern, email)
    
    # Function to validate phone number
    def validate_phone(phone):
        pattern = r'^[6-9]\d{9}$'
        return re.match(pattern, phone)
    
    # Main function
    def main():
        # st.title("User Registration")
    
        # Create a database connection
        conn = create_connection("dbs.db")
    
        if conn is not None:
            # Create users table if it doesn't exist
            conn.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY,
                         name TEXT NOT NULL,
                         password TEXT NOT NULL,
                         email TEXT NOT NULL UNIQUE,
                         phone TEXT NOT NULL);''')
    
            # User input fields
            
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your name:</label>
                """,
                unsafe_allow_html=True
            )
            name = st.text_input("")
            
    
            # Create the text input field and password field
            # name = st.text_input("Your name")
            
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your Password:</label>
                """,
                unsafe_allow_html=True
            )
            
            password = st.text_input("",type="password")
    
            
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your Confirm Password:</label>
                """,
                unsafe_allow_html=True
            )
            
            confirm_password = st.text_input(" ",type="password")
            
            # ------
    
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your Email ID:</label>
                """,
                unsafe_allow_html=True
            )
    
            email = st.text_input("  ")
            
            
            st.markdown(
                """
                <style>
                .custom-label {
                    font-size: 13px; /* Change the font size */
                    color: #000000;  /* Change the color */
                    font-weight: bold; /* Optional: make text bold */
                    display: inline-block; /* Make label inline with the input */
                    margin-right: 10px; /* Adjust the space between label and input */
                }
                .custom-input {
                    vertical-align: middle; /* Align input vertically with label */
                }
                </style>
                <label class="custom-label">Enter your Phone Number:</label>
                """,
                unsafe_allow_html=True
            )
            
            
            phone = st.text_input("   ")
    
            col1, col2 , col3 = st.columns(3)
    
            with col2:
                    
                aa = st.button("REGISTER")
                
                if aa:
                    
                    if password == confirm_password:
                        if not user_exists(conn, email):
                            if validate_email(email) and validate_phone(phone):
                                user = (name, password, email, phone)
                                create_user(conn, user)
                                st.success("User registered successfully!")
                            else:
                                st.error("Invalid email or phone number!")
                        else:
                            st.error("User with this email already exists!")
                    else:
                        st.error("Passwords do not match!")
                    
                    conn.close()
                    # st.success('Successfully Registered !!!')
                # else:
                    
                    # st.write('Registeration Failed !!!')     
            

    
    
      
    if __name__ == '__main__':
        main()


if navigation() == "log":
    
    st.markdown(f'<h1 style="color:#d35400 ;text-align: center;font-size:34px;font-family:verdana;">{"Green Options for milk packaging using intelligent packaging"}</h1>', unsafe_allow_html=True)
    st.write("-------------------------------------------")

    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:25px;font-family:Algerian;">{"Login Here !!!"}</h1>', unsafe_allow_html=True)
    
    
    
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('login.jpg')
    
    
    
    # Function to create a database connection
    def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)
        return conn
    
    # Function to create a new user
    def create_user(conn, user):
        sql = ''' INSERT INTO users(name, password, email, phone)
                  VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        return cur.lastrowid
    
    # Function to validate user credentials
    def validate_user(conn, name, password):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))
        user = cur.fetchone()
        if user:
            return True, user[1]  # Return True and user name
        return False, None
    
    # Main function
    def main():
        # st.title("User Login")
        # st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Login here"}</h1>', unsafe_allow_html=True)
    
    
        # Create a database connection
        conn = create_connection("dbs.db")
    
        if conn is not None:
            # Create users table if it doesn't exist
            conn.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY,
                         name TEXT NOT NULL,
                         password TEXT NOT NULL,
                         email TEXT NOT NULL UNIQUE,
                         phone TEXT NOT NULL);''')
    
            st.write("Enter your credentials to login:")
            name = st.text_input("User name")
            password = st.text_input("Password", type="password")
    
            col1, col2 = st.columns(2)
    
            with col1:
                    
                aa = st.button("Login")
                
                if aa:
    
    
            # if st.button("Login"):
                    is_valid, user_name = validate_user(conn, name, password)
                    if is_valid:
                        st.success(f"Welcome back, {user_name}! Login successful!")
                        
                        import subprocess
                        subprocess.run(['python','-m','streamlit','run','Prediction.py'])
                        
                        
                        
                    else:
                        st.error("Invalid user name or password!")
                        
            # with col2:
                      
            #       aa = st.button("Back")
                  
            #       if aa:
            #           import subprocess
            #           subprocess.run(['python','-m''streamlit','run','Student.py'])
                      # st.success('Successfully Registered !!!')          
    
            # Close the database connection
            conn.close()
        else:
            st.error("Error! cannot create the database connection.")
    
    if __name__ == '__main__':
        main()
       
