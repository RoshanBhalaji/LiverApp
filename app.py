from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('stacking_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

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

        # Print received data for debugging
        print("Received data:", data)

        # Validate input fields
        for field in FEATURE_ORDER:
            if field not in data:
                print(f"Missing field: {field}")
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
        prediction = model.predict(df_scaled)

        # Print prediction result for debugging
        print("Prediction result:", prediction)

        # Map prediction to result
        result = 'Disease Detected' if prediction[0] == 0 else 'No Disease Detected'

        return jsonify({'result': result})

    except Exception as e:
        print("Error occurred:", e)  # Log error details
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
