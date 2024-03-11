import streamlit as st
import cv2
import easyocr
import re
from ultralytics import YOLO
import pandas as pd
from fuzzywuzzy import fuzz
import imghdr
import warnings

# Streamlit app
st.title("ID Card Authentication using OCR")
# Ignore all warnings
# warnings.filterwarnings("ignore")

model = YOLO('best.pt')

def clean_string(input_list):
        # Join the list elements into a single string
        combined_string = ' '.join(input_list)
        # Remove special characters and punctuation marks
        cleaned_string = re.sub(r'[^\w\s]', '', combined_string)
        return cleaned_string
    
def fuzzy_match(template_text, extracted_text):
    # Perform fuzzy matching and return a similarity score
    return fuzz.token_set_ratio(template_text, extracted_text)
        
def load_csv_data(csv_path):
    # Load data from the CSV file using pandas
    return pd.read_csv(csv_path)
    

input_file_name = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]).name
st.write(input_file_name )
input_file_name = cv2.imread('t4.jpg')
if input_file_name:
    
    results = model.predict(source=input_file_name, show=False, save=False, conf=0.8, line_thickness=2, save_crop=False)
    # print(results)
    
    # Specify the path to the CSV file
    csv_path = "Aadhar.csv"
    
    #Load data from the CSV file
    csv_data = load_csv_data(csv_path)
    
    
    # Get the original image
    orig_img = results[0].orig_img
    # Initialize the OCR reader
    
    reader = easyocr.Reader(['en'])
    
    # Get the bounding boxes
    boxes = results[0].boxes.xyxy 
    # Initialize an empty dictionary to store the results
    class_texts = {}
    
    # Get the bounding boxes
    boxes = results[0].boxes.xyxy  # Modify this line as needed
    
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
    i=0
    for class_name in results[0].names.values():
        
        class_texts[class_name] = text_list[i]
    
        i=i+1
        
    #Iterate through rows in the CSV file for fuzzy matching
    for index, row in csv_data.iterrows():
        # Extract relevant information from the CSV row
        template_text = f"{row['Address']} {row['Aadhar Number']}" 
    
        # Perform fuzzy matching
        similarity_score = fuzz.partial_ratio(template_text, text_data) 
    
        # Set a threshold for similarity score
        threshold = 50
    
        # Perform ID card verification based on fuzzy matching
        if similarity_score >= threshold:
            st.success("ID card verified successfully!")
            st.write(f"Fuzzy Matching Score: {similarity_score}%")
            st.write(f"template: {template_text}")
            st.write(f"text data: {text_data}")
            break  
            
    if similarity_score < threshold:
        st.write("Unauthorized ID Card")
