import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download the necessary NLTK resources
# from nltk.tokenize import PunktTokenizer
nltk.download('punkt_tab') #nltk 3.8.2 only supports punkt_tab as punkt is an unsafe package.
nltk.download('stopwords')

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def preprocess(self, text):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        tokens = self.remove_stop_words(tokens)
        tokens = self.stem_tokens(tokens)
        return tokens

def preprocess_data(data):
    preprocessor = Preprocessor()
    processed_data = []

    for item in data:
        # Assuming each item is a dict containing 'text' key for review or name
        text = item.get('text', '') or item.get('name', '')
        processed_text = preprocessor.preprocess(text)
        processed_data.append({
            'original': text,
            'processed': processed_text
        })
    
    return processed_data

# if __name__ == "__main__":
#     # Example usage
#     example_data = [
#         {"text": "This restaurant is amazing! The food was delicious."},
#         {"text": "Not a good experience, the service was slow."}
#     ]
    
#     processed_example = preprocess_data(example_data)
#     for item in processed_example:
#         print(f"Original: {item['original']}")
#         print(f"Processed: {item['processed']}")

