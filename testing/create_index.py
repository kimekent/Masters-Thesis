"""
This script details the creation of BERT and OpenAI embedding indexes for the web support Q&As and web help articles.
The resulting indexes, stored as JSON files, are utilized by the testing_chatbot
(located at '\testing\testing_chatbot\testing_intent_less_chatbot.py'), which evaluates the retriever,
prompt & input size, and memory components.

The final chatbot implementation, as well as the test comparing the final intent-based Rasa chatbot with the
final intent-less chatbot, and the test comparing GPT-4 and GPT-3 text generation models, use the embeddings stored
in a Chroma vector database.

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.
"""

# Set path to root project directory and OpenAI API key
import sys
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis" # Change
testing_path = path + r'\testing'
sys.path.append(testing_path)

from testing_chatbot.testing_functions import open_file
import openai
import os
os.environ['OPENAI_API_KEY'] = open_file(path + '\openaiapikey.txt')
openai.api_key = os.getenv('OPENAI_API_KEY') # Add OpenAI API key to this .txt file

# Import libraries and functions
# Standard Libraries
import pandas as pd
import json
import os

# Libraries to create embeddings
from sentence_transformers import SentenceTransformer

# Own functions
from testing_chatbot.testing_functions import replace_links_with_placeholder

# 1. Setup--------------------------------------------------------------------------------------------------------------
webhelp_folder = testing_path + r'\testing_data\webhelp_articles'
websupport_questions_path = testing_path + r'\testing_data\websupport_train_dataset.csv'

# Load the CSV containing questions and answers
testing_data_set = pd.read_csv(websupport_questions_path)
questions = testing_data_set["Beschreibung"].tolist()
questions = [replace_links_with_placeholder(i) for i in questions]
answer = testing_data_set["Lösungsbeschreibung"].tolist()
incident_ids = testing_data_set["Incident-ID"].tolist()
intent = testing_data_set["intent"].tolist()


# 2. Initialize the models used for embedding the webhelp articles and websupport Q&A-----------------------------------

# Initialize the model using BERT embeddings
model_name = 'sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base'
context_model = SentenceTransformer(model_name) # Initialize the SentenceTransformer model

# Define the model using OpenAI embeddings
def gpt3_embedding(content, engine="text-embedding-ada-002"):
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


# 3. Create embeddings and store as JSON file---------------------------------------------------------------------------

# Create and store BERT embeddings--------------------------------------------------------------------------------------
# Create a list to store the JSON entries
index_data = []

# Process webhelp articles
for filename in os.listdir(webhelp_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(webhelp_folder, filename), 'r', encoding='utf-8') as article_file:
            content = article_file.read()
            embedding = context_model.encode(content)
            entry = {
                "context": content,
                "metadata": {"source": "webhelp_article", "Link": content.split("Link: ")[1].strip()},
                "vector": embedding.tolist()
            }
            index_data.append(entry)

# Process websupport questions and answers
for i in range(len(questions)):
    entry = {
        "context": questions[i],
        "metadata": {
            "source": "websupport_question",
            "question_id": incident_ids[i],
            "answer": answer[i],
            "intent": intent[i]
        },
        "vector": context_model.encode(questions[i]).tolist()
    }
    index_data.append(entry)

# Save the index data to a JSON file
with open(path + r'\indexes\BERT_index.json', 'w') as json_file:
    json.dump(index_data, json_file, indent=2)

# Create and store OpenAI embeddings------------------------------------------------------------------------------------
# Initialize an empty list to store the index data
index_data = []

# Process web help articles
for filename in os.listdir(webhelp_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(webhelp_folder, filename), 'r', encoding='utf-8') as article_file:
            content = article_file.read()
            embedding = gpt3_embedding(content)
            entry = {
                "context": content,
                "metadata": {"source": "webhelp_article", "Link": content.split("Link: ")[1].strip()},
                "vector": embedding
            }
            index_data.append(entry)

# Process websupport questions and answers
for i in range(len(questions)):
    content = questions[i]
    embedding = gpt3_embedding(content)
    entry = {
        "context": content,
        "metadata": {
            "source": "websupport_question",
            "question_id": incident_ids[i],
            "answer": answer[i],
            "intent": intent[i]
        },
        "vector": embedding
    }
    index_data.append(entry)

# Save the index data to a JSON file
with open(path + r'\indexes\OpenAI_index.json', 'w') as json_file:
    json.dump(index_data, json_file, indent=2)