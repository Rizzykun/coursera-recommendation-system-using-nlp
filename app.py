from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Initialize Porter Stemmer
ps = PorterStemmer()

# Load data from pickle files
print("Loading data from pickle files...")
try:
    # Load courses DataFrame
    courses_list = pickle.load(open('courses.pkl', 'rb'))
    print("Successfully loaded courses.pkl")
    
    # Load similarity matrix
    similarity = pickle.load(open('course_list.pkl', 'rb'))
    print("Successfully loaded course_list.pkl")
except Exception as e:
    print(f"Error loading pickle files: {e}")
    print("Please ensure the pickle files are in the correct location.")
    exit(1)

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

@app.route('/courses')
def get_courses():
    """Return the list of all course names"""
    return jsonify(courses_list['course_name'].tolist())

@app.route('/recommend')
def get_recommendations():
    """Get recommendations based on a course name"""
    course = request.args.get('course', '')
    if not course:
        return jsonify({"error": "No course provided"}), 400
    
    try:
        recommendations = recommend(course)
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Make sure NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)