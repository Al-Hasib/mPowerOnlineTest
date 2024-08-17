import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import *
import joblib


test_path = Path("./ML Engineer/test.csv")
submission_path = Path("./ML Engineer/sample_submission.csv")
train_path = Path("./ML Engineer/train.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission_df = pd.read_csv(submission_path)

submission_df = submission_df.drop('class', axis=1)

currency_symbols = r'[\$\£\€\¥\₹\¢\₽\₩\₪]'  
text_cleaner = TextCleaner(currency_symbols)
train['clean_text'] = train['email'].apply(lambda x: text_cleaner.clean_text(x))

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit(train['clean_text'])


test['clean_text'] = test['email'].apply(lambda x: text_cleaner.clean_text(x))
print(test.head())

X = vectorizer.transform(test['clean_text'])
print(X.shape)

# Load the model from the file
loaded_model = joblib.load('email_detection_model.pkl')

predictions = loaded_model.predict(X)


submission_df['class'] = predictions
submission_df['class'] = submission_df['class'].map({1: 'spam', 0: 'not_spam'})

print(submission_df.head())

# Save DataFrame to a CSV file
submission_df.to_csv('submission_md_abdullah_al_hasib.csv', index=False)

print("Evaluation completed..")