from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import google.generativeai as genai
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('stacking_classifier_model.pkl', 'rb') as model_file:
    ml_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask app
app = Flask(__name__)

# Initialize LabelEncoder for Gender
gender_encoder = LabelEncoder()
gender_encoder.fit(['Female', 'Male'])  # Fit encoder with gender categories

# Define feature order (match the order used during training)
FEATURE_ORDER = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
    'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
]

# GEMINI API Configuration
GEMINI_API_KEY = "AIzaSyDqf8ua1UYQzMo9XutHzIjl_I60Evij_U8"

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the content type is JSON or form data
        if request.is_json:
            data = request.get_json()  # Handle JSON data
        else:
            data = request.form  # Handle form data

        # Validate input fields
        for field in FEATURE_ORDER:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Prepare input in the correct order
        inputs = {
            'Gender': data['Gender'],
            'Age': float(data['Age']),
            'Total_Bilirubin': float(data['Total_Bilirubin']),
            'Direct_Bilirubin': float(data['Direct_Bilirubin']),
            'Alkaline_Phosphotase': float(data['Alkaline_Phosphotase']),
            'Alamine_Aminotransferase': float(data['Alamine_Aminotransferase']),
            'Aspartate_Aminotransferase': float(data['Aspartate_Aminotransferase']),
            'Total_Protiens': float(data['Total_Protiens']),
            'Albumin': float(data['Albumin']),
            'Albumin_and_Globulin_Ratio': float(data['Albumin_and_Globulin_Ratio']),
        }

        # Create DataFrame with the correct feature order
        df = pd.DataFrame([inputs])[FEATURE_ORDER]

        # Encode categorical feature (Gender)
        df['Gender'] = gender_encoder.transform(df['Gender'])

        # Apply log transformation to numerical columns
        numerical_cols = df.columns.difference(['Gender'])
        df[numerical_cols] = np.log1p(df[numerical_cols])

        # Scale features using pre-trained scaler
        df_scaled = scaler.transform(df)

        # Make prediction
        prediction = ml_model.predict(df_scaled)

        # Map prediction to result
        result = 'Disease Detected' if prediction[0] == 0 else 'No Disease Detected'

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/chat')
def chat_home():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message')  # Get the user input message

    # If no message is provided, return an error
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Start a chat session
        chat_session = gemini_model.start_chat(history=[])

        # Construct the prompt
        prompt = """
        You are a medical assistant. Answer the user's question with:
        
        1. Clear medical advice.
        2. Tips for managing the condition, including lifestyle changes.
        3. Research-backed natural remedies.
        4. Remind to consult a healthcare provider for personalized advice.

        Disclaimer: "This information is for educational purposes only. Please consult a healthcare provider for personalized medical advice."
        """
        
        prompt += f' The user asked: "{user_message}".'

        # Send the message to the Gemini API
        response = chat_session.send_message(prompt)

        # Return the AI response
        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
