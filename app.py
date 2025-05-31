from flask import Flask, request, render_template
import pickle
import joblib
import re
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

app = Flask(__name__)

# Load models
lr_model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
lstm_model = tf.keras.models.load_model('lstm_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess review text
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    words = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')
    model_type = request.form.get('model_type', 'lr')

    cleaned_review = preprocess_text(review)

    if model_type == 'lr':
        features = tfidf.transform([cleaned_review])
        prediction = lr_model.predict(features)[0]
        proba = lr_model.predict_proba(features)[0][prediction]
    else:
        sequence = tokenizer.texts_to_sequences([cleaned_review])
        padded = pad_sequences(sequence, maxlen=200)
        proba_val = lstm_model.predict(padded)[0][0]
        prediction = 1 if proba_val > 0.5 else 0
        proba = proba_val if prediction == 1 else 1 - proba_val

    sentiment = 'positive' if prediction == 1 else 'negative'

    return render_template(
        'index.html',
        sentiment=sentiment,
        confidence=proba,
        model='Logistic Regression' if model_type == 'lr' else 'LSTM'
    )

if __name__ == '__main__':
    app.run(debug=True)
