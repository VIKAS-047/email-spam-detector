from flask import Flask, request, render_template
import os
from joblib import load

app = Flask(__name__)  # Use __name__ with double underscores

# Load the saved model and vectorizer from the model pipeline
model_path = os.path.join('models', 'best_model.pkl')
best_model = load(model_path)  # This loads the entire pipeline including the vectorizer and classifier

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    
    # Vectorize and predict using the loaded model pipeline
    prediction = best_model.predict([email_text])[0]
    result = "Spam" if prediction == 'spam' else "Not Spam"
    
    return render_template('index.html', result=result)

if __name__ == "__main__":  # Use __name__ with double underscores
    app.run(debug=True)
