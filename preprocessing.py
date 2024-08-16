import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextCleaner:
    def __init__(self, currency_symbols, stop_words=None, lemmatizer=None):
        self.currency_symbols = currency_symbols
        
        if stop_words is None:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = stop_words
        
        if lemmatizer is None:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = lemmatizer
    
    def remove_punctuation(self,text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def clean_text(self, text):
        text = text.lower()
        
        # Replace all occurrences of currency symbols with the word 'currency'
        text = re.sub(self.currency_symbols, 'currency', text)
        
        # Remove punctuation
        text = self.remove_punctuation(text)
        
        # Remove HTML tags
        text = re.compile('<.*?>').sub('', text)
        
        # Remove underscores
        text = text.replace('_', '')
        
        # Remove remaining non-word characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove digits
        text = re.sub(r'\d', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        
        # Lemmatize the text
        text = ' '.join(self.lemmatizer.lemmatize(word) for word in text.split())
        
        return text

