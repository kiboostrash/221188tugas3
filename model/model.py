# model/train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv('data/tugas1.csv')

# Features (menggunakan TITLE dan ABSTRACT sebagai input)
X = data[['TITLE', 'ABSTRACT']]
# Target (menggunakan salah satu kolom bidang, misalnya "Computer Science")
y = data['Computer Science']

# Preprocessing sederhana: menggabungkan TITLE dan ABSTRACT
X['combined_text'] = X['TITLE'] + " " + X['ABSTRACT']

# Vectorization: Mengubah teks menjadi vektor numerik
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_vectors = vectorizer.fit_transform(X['combined_text']).toarray()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and vectorizer
with open('model/model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

print("Model saved successfully!")
