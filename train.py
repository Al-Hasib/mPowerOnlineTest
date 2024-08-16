import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from preprocessing import *
import joblib

train_path = Path("./ML Engineer/train.csv")
df = pd.read_csv(train_path)

encoder = LabelEncoder()
# Fit and transform the labels to numeric values
df['target'] = encoder.fit_transform(df['class'])

currency_symbols = r'[\$\£\€\¥\₹\¢\₽\₩\₪]'  
text_cleaner = TextCleaner(currency_symbols)

df['clean_text'] = df['email'].apply(lambda x: text_cleaner.clean_text(x))

print(df.head())

vectorizer = CountVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['target']
print(X.shape, y.shape)

# Initialize the classifier
lr_classifier = LogisticRegression()

# Train the model
lr_classifier.fit(X, y)

print("Training Completed")

# Save the model to a file
joblib.dump(lr_classifier, 'email_detection_model.pkl')
