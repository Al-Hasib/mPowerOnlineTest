import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import string
from sklearn.model_selection import train_test_split
from preprocessing import *


train_path = Path("./ML Engineer/train.csv")
df = pd.read_csv(train_path)

encoder = LabelEncoder()
# Fit and transform the labels to numeric values
df['target'] = encoder.fit_transform(df['class'])

currency_symbols = r'[\$\£\€\¥\₹\¢\₽\₩\₪]'  
text_cleaner = TextCleaner(currency_symbols)

df['clean_text'] = df['email'].apply(lambda x: text_cleaner.clean_text(x))

print(df.head())

