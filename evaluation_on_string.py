import joblib
from preprocessing import *
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pathlib import Path

text = input("Enter the Email: \n")

train_path = Path("./ML Engineer/train.csv")
df = pd.read_csv(train_path)
currency_symbols = r'[\$\£\€\¥\₹\¢\₽\₩\₪]'  
text_cleaner = TextCleaner(currency_symbols)
df['clean_text'] = df['email'].apply(lambda x: text_cleaner.clean_text(x))

vectorizer = CountVectorizer(max_features=10000)
X = vectorizer.fit(df['clean_text'])


clean_text = str(text_cleaner.clean_text(text))

print(f"\nThe clean text is : {clean_text}")

y = vectorizer.transform([clean_text])

# Load the model from the file
loaded_model = joblib.load('email_detection_model.pkl')

predictions = loaded_model.predict(y)[0]

predictions = "spam" if predictions=='1' else "not_spam"

print(f"\nThe prediction is : {predictions}")