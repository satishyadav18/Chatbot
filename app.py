from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# Load your trained machine learning model
model = pickle.load(open('mod.pkl','rb'))
le = pickle.load(open('le.pkl','rb'))

# List to store user inputs
user_inputs = []

# Define the sequence of questions
questions = [
    "What is the temperature of the region?",
    "What is the humidity level of the region?",
    "What is the soil moisture level?",
    "What is the soil type?",
    "What is the type of crop you wish to grow?",
    "What is the nitrogen content of the soil?",
    "What is the potassium content of the soil?",
    "What is the phosphorus content of the soil?"
]

# Index to track the current question
current_question_index = 0

# List to store the conversation history
conversation_history = []

@app.route('/')
def home():
    global current_question_index, conversation_history
    current_question_index = 0  # Reset when starting the conversation
    conversation_history = []  # Reset conversation history
    return render_template('index.html', question=questions[current_question_index], conversation=conversation_history)

@app.route('/chat', methods=['POST'])
def chat():
    global current_question_index, user_inputs, conversation_history
    
    # Get the user's response
    user_response = request.form['response']
    
    # Store the user's response
    user_inputs.append(user_response)
    
    # Add the question and response to the conversation history
    conversation_history.append({"question": questions[current_question_index], "response": user_response})

    # Increment the question index
    current_question_index += 1
    
    # If we have all inputs, make a prediction
    if current_question_index == len(questions):
        # Prepare the input for the model
        features = np.array(user_inputs).reshape(1, -1)
        
        # Create a DataFrame from user inputs
        new_data = pd.DataFrame({
            "Temperature": [features[0][0]],
            "Humidity": [features[0][1]],
            "Moisture": [features[0][2]],
            "Soil Type": [features[0][3]],
            "Crop Type": [features[0][4]],
            "Nitrogen": [features[0][5]],
            "Potassium": [features[0][6]],
            "Phosphorous": [features[0][7]]
        })

        # Predict with the model
        prediction = model.predict(new_data)
        fertilizer = le.inverse_transform(prediction)[0]
        
        # Add the prediction to the conversation history
        conversation_history.append({"question": "The suitable fertilizer is:", "response": fertilizer})
        
        # Clear the inputs for a new conversation
        user_inputs = []
        current_question_index = 0
        
        return render_template('index.html', conversation=conversation_history, question=None)
    
    # Otherwise, continue asking questions
    next_question = questions[current_question_index]
    return render_template('index.html', question=next_question, conversation=conversation_history)


