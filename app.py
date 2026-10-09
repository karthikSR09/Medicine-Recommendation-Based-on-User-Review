from flask import Flask, render_template, request, redirect, session, flash
import os
import joblib #slml
import pandas as pd
import re
import numpy as np
import nltk
import csv
from nltk.stem import WordNetLemmatizer #bfr
from nltk.corpus import stopwords
from bs4 import BeautifulSoup #lws
from sklearn import metrics #eml
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary nltk data
nltk.download('wordnet')
nltk.download('vader_lexicon')
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# CSV path for storing users
USERS_FILE = 'users.csv'
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['username', 'email', 'phone', 'password'])

# Model and vectorizer paths
MODEL_PATH = 'model/passmodel.pkl'
TOKENIZER_PATH = 'model/tfidfvectorizer.pkl'
DATA_PATH = 'drugs/drugsComTrain.csv'

vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/register_validation', methods=['POST'])
def register_validation():
    username = request.form.get('username')
    email = request.form.get('email')
    phone = request.form.get('phone')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    # Check if passwords match
    if password != confirm_password:
        flash("Passwords do not match.")
        return redirect('/register')

    # Check if username or email already exists
    with open(USERS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['username'] == username:
                flash("Username already exists.")
                return redirect('/register')
            if row['email'] == email:
                flash("Email already exists.")
                return redirect('/register')

    # Save new user details to CSV
    file_exists = os.path.isfile(USERS_FILE)
    with open(USERS_FILE, 'a', newline='') as f:
        fieldnames = ['username', 'email', 'phone', 'password']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists or os.stat(USERS_FILE).st_size == 0:
            writer.writeheader()  # write header if file is empty

        writer.writerow({
            'username': username,
            'email': email,
            'phone': phone,
            'password': password
        })

    flash("Registration successful. Please log in.")
    return redirect('/')


@app.route('/login_validation', methods=['POST'])
def login_validation():
    username = request.form.get('username')
    password = request.form.get('password')

    with open(USERS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['username'] == username and row['password'] == password:
                session['user_id'] = username
                return redirect('/index')

    flash("Invalid credentials.")
    return redirect('/')


@app.route("/logout")
def logout():
    session.clear()
    return redirect('/')


@app.route('/index')
def index():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/')


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        raw_text = request.form['rawtext']

        if raw_text != "":
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]

            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]

            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_cond, df)

            sid = SentimentIntensityAnalyzer()
            score = sid.polarity_scores(str(clean_text))['compound']
            if score > 0:
                labeli = 'This sentence is positive'
            elif score == 0:
                labeli = 'This sentence is neutral'
            else:
                labeli = 'This sentence is negative'

            return render_template('predict.html',
                                   rawtext=raw_text,
                                   result=predicted_cond,
                                   top_drugs=top_drugs,
                                   variable=labeli)

        flash("No input text received.")
        return redirect('/index')


def cleanText(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stop]
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmitize_words)


def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'],
                                                                              ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst



if __name__ == "__main__":
    app.run(debug=False, host="localhost", port=8000)
