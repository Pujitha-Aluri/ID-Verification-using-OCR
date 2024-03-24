import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sqlite3
from flask_bcrypt import bcrypt

import cv2
import easyocr
import re
from ultralytics import YOLO
import pandas as pd
from fuzzywuzzy import fuzz
import imghdr
import warnings


# Specify the path to the CSV file
csv_path = "Aadhar.csv"


# Streamlit app
st.title("ID Card Authentication using OCR")
# Ignore all warnings
# warnings.filterwarnings("ignore")

model = YOLO('best.pt')

def clean_string(input_list):
        # Join the list elements into a single string
        combined_string = ' '.join(input_list)
        # # Remove special characters and punctuation marks
        # cleaned_string = re.sub(r'[^\w\s]', '', combined_string)
        return combined_string
    
def fuzzy_match(template_text, extracted_text):
    # Perform fuzzy matching and return a similarity score
    return fuzz.token_set_ratio(template_text, extracted_text)
        
def load_csv_data(csv_path):
    # Load data from the CSV file using pandas
    return pd.read_csv(csv_path)
    

#Load data from the CSV file
csv_data = load_csv_data(csv_path)

# Initialize SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def signup(username, password):
    # Check if user already exists
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    if c.fetchone():
        return False
    else:
        # Hash the password and store the user
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return True

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def login(username, password):
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if user and verify_password(password, user[1]):
        return True
    else:
        return False





if __name__=='__main__':
    st.title('ID Card Authorization')


    auth_status = st.session_state.get('auth_status', None)
    if auth_status == "logged_in":
        st.success(f"Welcome {st.session_state.username}!")
        st.header('Select an id card to verify')

        #####
        input_file_name = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]).name
        st.write(input_file_name)
        # input_file_name = 't4.jpg'
        if input_file_name:
            
            results = model.predict(source=input_file_name, show=False, save=False, conf=0.8, line_thickness=2, save_crop=False)
            # print(results)
            
            
            
            # Get the original image
            orig_img = results[0].orig_img
            # Initialize the OCR reader
            reader = easyocr.Reader(['en'])
            
            # Get the bounding boxes
            boxes = results[0].boxes.xyxy 
            # Initialize an empty dictionary to store the results
            class_texts = {}
            
            # Get the bounding boxes
            boxes = results[0].boxes.xyxy 
            
            # Iterate over each box
            text_list=[]
            for i, box in enumerate(boxes):
                # Convert the box coordinates to integers
                x1, y1, x2, y2 = map(int, box)
                cropped_img = orig_img[y1:y2, x1:x2]
                text = reader.readtext(cropped_img,detail=0)
                text = clean_string(text)
                text_list.append(text)
                
            text_data = clean_string(set(text_list))
            text_data = ''.join(filter(lambda x: x.isdigit(), text_data))
            # text_data=text_data[:11]
            
            # # Iterate over each detected class
            # i=0
            # for class_name in results[0].names.values():
                
            #     class_texts[class_name] = text_list[i]
            
            #     i=i+1
                
            #Iterate through rows in the CSV file for fuzzy matching
            for index, row in csv_data.iterrows():
                # Extract relevant information from the CSV row
                template_text = f"{row['Address']} {row['Aadhar Number']} {row['Government']}" 
            
                # Perform fuzzy matching
                similarity_score = fuzz.partial_ratio(template_text, text_data) 
            
                # Set a threshold for similarity score
                threshold = 50
            
                # Perform ID card verification based on fuzzy matching
                if similarity_score >= threshold:
                    st.success("ID card verified successfully!")
                    # st.write(f"Fuzzy Matching Score: {similarity_score}%")
                    st.write(f"template: {template_text}")
                    # st.write(f"text data: {text_data}")
                    break  
                    
            if similarity_score < threshold:
                st.write("Unauthorized ID Card")


    elif auth_status == "login_failed":
        st.error("Login failed. Please check your username and password.")
        auth_status = None
    elif auth_status == "signup_failed":
        st.error("Signup failed. Username already exists.")
        auth_status = None
    # Login/Signup form
    if auth_status is None or auth_status == "logged_out":
        form_type = st.radio("Choose form type:", ["Login", "Signup"])

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if form_type == "Login":
            if st.button("Login"):
                if login(username, password):
                    st.session_state.auth_status = "logged_in"
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.session_state.auth_status = "login_failed"
                    st.rerun()
        else:  # Signup
            if st.button("Signup"):
                if signup(username, password):
                    st.session_state.auth_status = "logged_in"
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.session_state.auth_status = "signup_failed"
                    st.rerun()

    # Logout button
    if auth_status == "logged_in":
        if st.button("Logout"):
            st.session_state.auth_status = "logged_out"
            del st.session_state.username
            st.rerun()