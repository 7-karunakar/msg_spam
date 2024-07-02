from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('model/spam_modelsvm.pkl')
vectorizer = joblib.load('model/vectorizersvm.pkl')

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']
        
        # Transform the message using the loaded vectorizer
        x = vectorizer.transform([message])
        
        # Predict using the loaded model
        prediction = model.predict(x)
        
        # Map prediction to human-readable label
        if prediction == 0:
            prediction_label = "ham"
        else:
            prediction_label = "spam"

        # Render the result.html with prediction
        return render_template('result.html', message=message, prediction=prediction_label)
    
    # Render the index.html for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
