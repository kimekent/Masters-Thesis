"""
This script includes the code necessary to identify keywords associated with each intent.
The results from this process are utilized in the 'adjust similarity score' function of the retriever.
This function scans for the identified keywords within questions, and if such a keyword is detected,
it enhances the similarity score for the intent linked to that keyword.

To run this script update the 'path' variable to the project directory path.
"""

# Set directory path
path = r'C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis\testing' # Change

# Import libraries and functions
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from testing_functions import open_file


# Download the stopwords from NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Set of German stopwords
additional_stopwords = set(open_file(path + r'\additional_german_stopwords.txt'))
german_stopwords = set(stopwords.words('german')).union(additional_stopwords)

# Get the web support ticketing dataset
websupport_question_df = pd.read_csv(path + r'\testing_data\cleaned_websupport_questions_with_intents_utf-8.csv')

# Function to replace URLs with a placeholder
def replace_links_with_placeholder(text):
    url_pattern = r'https://\S+'
    return re.sub(url_pattern, '<<link>>', text)

# Function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace URLs with a placeholder
    text = replace_links_with_placeholder(text)
    # Remove all numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = nltk.word_tokenize(text, language='german')
    # Remove stopwords
    tokens = [token for token in tokens if token not in german_stopwords]

    # Stemming
    #stemmer = GermanStemmer()
    #stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Rejoin tokens into a string
    return ' '.join(tokens)

# Preprocess text
websupport_question_df['processed_questions'] = websupport_question_df['Beschreibung'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(websupport_question_df['processed_questions'])

# Create a DataFrame for TF-IDF vectors and map it to intents
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df['intent'] = websupport_question_df['intent']

# Analyzing TF-IDF scores for each intent
for intent in tfidf_df['intent'].unique():
    print(f'Top words for {intent}:')
    temp_df = tfidf_df[tfidf_df['intent'] == intent].drop('intent', axis=1)
    mean_scores = temp_df.mean(axis=0).sort_values(ascending=False)
    print(mean_scores.head(10))  # Prints top 5 words