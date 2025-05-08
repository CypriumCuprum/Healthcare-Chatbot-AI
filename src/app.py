import os
import sys
from flask import Flask, render_template, request, jsonify
from chatbot.gemini_client import GeminiClient
from model.predictor import HealthPredictor
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Create static and templates directories if they don't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

# Initialize the health predictor and Gemini client
gemini_client = GeminiClient(api_key=os.getenv("GEMINI_API_KEY"))
health_predictor = HealthPredictor()

# Check if model needs to be trained
if not health_predictor.is_model_loaded():
    print("Model not found. Please run 'python src/data/generate_data.py' and 'python src/model/train_model.py' first.")

@app.route('/')
def index():
    """Render the main page."""
    symptom_fields = [
        {'name': 'ho', 'label': 'Ho'},
        {'name': 'sot', 'label': 'Sốt'},
        {'name': 'dau_hong', 'label': 'Đau họng'},
        {'name': 'dau_bung', 'label': 'Đau bụng'},
        {'name': 'mat_khuu_giac', 'label': 'Mất khứu giác'},
        {'name': 'kho_tho', 'label': 'Khó thở'}
    ]
    return render_template('index.html', symptom_fields=symptom_fields)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle symptom prediction requests."""
    if not health_predictor.is_model_loaded():
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        })
    
    # Get symptoms from form data
    symptoms = {}
    for field in health_predictor.feature_names:
        # Get value from form (0 or 1)
        symptoms[field] = int(request.form.get(field, '0'))
    
    # Make prediction
    predicted_disease, probabilities = health_predictor.predict(symptoms)
    
    # Get description and recommendations
    description = health_predictor.get_disease_description(predicted_disease)
    recommendations = health_predictor.get_recommendations(predicted_disease)
    
    # Return the prediction results
    return jsonify({
        'success': True,
        'disease': predicted_disease,
        'description': description,
        'recommendations': recommendations,
        'probabilities': probabilities
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Handle health question requests to Gemini."""
    if not gemini_client.is_configured():
        return jsonify({
            'success': False,
            'error': 'Gemini API is not configured. Please provide an API key.'
        })
    
    # Get the question from the request
    question = request.form.get('question', '')
    
    if not question:
        return jsonify({
            'success': False,
            'error': 'No question provided.'
        })
    
    # Ask Gemini for an answer
    response = gemini_client.ask_health_question(question)
    
    # Return the answer
    return jsonify({
        'success': True,
        'question': question,
        'answer': response
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)