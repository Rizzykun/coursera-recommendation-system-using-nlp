# <==== Importing Dependencies ====>

import os
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Add this after your imports
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def recommend(course):
    # Check if course exists in dataset
    if course in courses_list['course_name'].values:
        # Existing logic for courses in the dataset
        index = courses_list[courses_list['course_name'] == course].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_course_names = []
        for i in distances[1:7]:
            course_name = courses_list.iloc[i[0]].course_name
            recommended_course_names.append(course_name)
        return recommended_course_names
    else:
        # New logic for courses not in the dataset
        # Process the input text similar to how the model was trained
        input_text = course.lower()
        input_text = stem(input_text)
        
        # Create a vector for the input text using the same approach
        cv = CountVectorizer(max_features=5000, stop_words='english')
        course_vectors = cv.fit_transform(courses_list['tags']).toarray()
        
        # Add the input text to be transformed
        all_text = list(courses_list['tags'])
        all_text.append(input_text)
        
        # Refit and transform
        all_vectors = cv.fit_transform(all_text).toarray()
        
        # Calculate similarity between input and all courses
        input_vector = all_vectors[-1].reshape(1, -1)
        course_vectors = all_vectors[:-1]
        similarities = cosine_similarity(input_vector, course_vectors)[0]
        
        # Get top 6 most similar courses
        course_indices = similarities.argsort()[-6:][::-1]
        recommended_course_names = [courses_list.iloc[idx].course_name for idx in course_indices]
        return recommended_course_names
    
st.markdown("<h2 style='text-align: center; color: blue;'>Coursera Course Recommendation System</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Find similar courses from a dataset of over 3,000 courses from Coursera!</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: black;'>Web App created by Sagar Bapodara</h4>", unsafe_allow_html=True)

# Option to select from existing courses

courses_list = pd.read_csv("path_to_your_courses_file.csv")

st.subheader("Option 1: Select an existing course")
course_list = courses_list['course_name'].values
selected_course = st.selectbox(
    "Choose a course you like:",
    courses_list
)

# Option to enter any course name
st.subheader("Option 2: Enter any course name/topic")
custom_course = st.text_input("Enter a course name or topic of interest:")

if st.button('Show Recommended Courses'):
    # Determine which input to use
    course_to_use = custom_course if custom_course else selected_course
    
    st.write(f"Recommended Courses based on: '{course_to_use}'")
    recommended_course_names = recommend(course_to_use)
    
    # Display recommendations
    for i, course in enumerate(recommended_course_names, 1):
        st.write(f"{i}. {course}")
    
    st.text(" ")
    st.markdown("<h6 style='text-align: center; color: red;'>Copyright reserved by Coursera and Respective Course Owners</h6>", unsafe_allow_html=True)
