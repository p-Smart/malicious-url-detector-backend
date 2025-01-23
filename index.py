from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Initialize Flask app
app = Flask(__name__)

CORS(app)

def load_file(file_name):
    """Utility function to load a pickled file."""
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def test_url_with_saved_model(url, model_name):
    """
    Predict if a URL is malicious using a specified model.

    Args:
        url (str): The URL to test.
        model_name (str): The name of the model to use.

    Returns:
        dict: A response dictionary with the prediction results.
    """
    try:
        # Load necessary files
        model = load_file(f'{model_name}_model.pkl')
        vectorizer = load_file('vectorizer.pkl')
        label_encoder = load_file('label_encoder.pkl')

        # Vectorize the URL and make a prediction
        url_vectorized = vectorizer.transform([url])
        prediction = model.predict(url_vectorized)
        predicted_label = label_encoder.inverse_transform(prediction)

        return {
            "success": True,
            "data": {"prediction": predicted_label[0]},
            "message": "Prediction successful"
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "data": None,
            "message": f"File not found: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": f"An error occurred: {e}"
        }

@app.route('/check-url', methods=['POST'])
def check_url():
    """
    API endpoint to check if a URL is malicious.

    Expects a JSON payload with the fields:
    - 'url': The URL to check.
    - 'model': The model to use for the prediction.

    Returns:
        JSON response with the prediction results.
    """
    data = request.get_json()

    # Validate payload
    if not data or 'url' not in data or 'model' not in data:
        return jsonify({
            "success": False,
            "data": None,
            "message": "Invalid payload"
        }), 400

    # Extract input data
    url = data['url']
    model_name = data['model']

    # Process the prediction
    response = test_url_with_saved_model(url, model_name)
    return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)
