import pickle

# Function to test a URL using a saved model
def test_url_with_saved_model(url, model_name):
    try:
        # Load the saved model
        with open(f'{model_name}_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"{model_name} model loaded successfully!")
        
        # Load the vectorizer
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        
        # Load the label encoder
        with open('label_encoder.pkl', 'rb') as label_encoder_file:
            label_encoder = pickle.load(label_encoder_file)
        
        # Vectorize the input URL
        url_vectorized = vectorizer.transform([url])
        
        # Predict using the loaded model
        prediction = model.predict(url_vectorized)
        
        # Decode the predicted label
        predicted_label = label_encoder.inverse_transform(prediction)
        
        print(f"Prediction for URL '{url}': {predicted_label[0]}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure the model, vectorizer, and label encoder files are available.")
    except Exception as e:
        print(f"An error occurred: {e}")


# To run and test the code (two parameters; the url, and the model to use)

test_url_with_saved_model("http://9779.info/%E5%B9%BC%E5%84%BF%E7%B2%BD%E5%8F%B6%E8%B4%B4%E7%94%BB/", "svm")