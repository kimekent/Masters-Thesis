"""
This file was used to create the Chroma vector database, that the final version of the intent-less uses
(located at '\intent-less_chatbot\chatbot.py').
It includes the text, embeddings and metadata of all web support Q&As and web help articles.

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.
"""


# 1. Set up-------------------------------------------------------------------------------------------------------------
# Set path to project directory and define OpenAI API key
import sys
path = r'C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis'  # Change
intent_less_path = path + r'\intent-less_chatbot'
sys.path.append(intent_less_path)

from functions import open_file
import openai
import os
os.environ['OPENAI_API_KEY'] = open_file(path + '\openaiapikey.txt')
openai.api_key = os.getenv('OPENAI_API_KEY')  # Add OpenAI API key to this .txt file

# Import packages
import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from functions import replace_links_with_placeholder

# Set the persist_directory, where the embeddings are stored on disk
persist_directory = intent_less_path + r'\webhelp_and_websupport_vector_db'
# Path to CSV file containing the web support question and answers
websupport_questions = intent_less_path + r'\data\cleaned_websupport_questions_with_intents_utf-8.csv'

# Embeddings that are used for the Chroma Database
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# Read questions and answers from the CSV file
df = pd.read_csv(websupport_questions, encoding='utf-8')
filtered_df = df[df['Anfrageart'] == 'Dokumentation nicht gelesen']
questions = filtered_df['Beschreibung'].tolist()
questions = [replace_links_with_placeholder(i) for i in questions]
incident_ids = filtered_df['Incident-ID'].tolist()
answers = filtered_df['Lösungsbeschreibung'].tolist()
intents = filtered_df['intent'].tolist()

# 2. Create Chroma vector database--------------------------------------------------------------------------------------
# Create langchain Documents from questions and answers
docs = []
for i in range(len(questions)):
    question = questions[i]
    answer = answers[i]
    incident_id = incident_ids[i]
    intent = intents[i]
    document = Document(
        page_content=question,  # Questions are used as page content
        metadata={
            'Source': 'websupport question',
            'Answer': answer,
            # This corresponds to the ID of the question from the web support dataset
            'Incident_ID': incident_id,
            'intent': intent,
            'id': i  # Add an 'id' field with the current index i as its value
        }
    )
    docs.append(document)

# Embed and store the texts
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

docs = []
i = len(questions) + 1 # Initialize the 'i' variable outside the loop
for filename in os.listdir(intent_less_path + '\data\chunked_webhelp_articles'):
    if filename.endswith('.txt'):
        with open(os.path.join(intent_less_path + '\data\chunked_webhelp_articles', filename), 'r', encoding='utf-8') as file:
            content = file.read()
            document = Document(
                page_content=content,
                metadata={
                    'Source': 'webhelp-article',
                    'Key': content.split('Key: ')[1].split('\n')[0].strip(),
                    'Link': content.split('Link: ')[1].strip(),
                    'id': i  # Set the 'id' to the value of 'i'
                }
            )
            docs.append(document)
            i += 1  # Increment 'i' for the next document

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

# 3. Check to see if database exists and was configured properly--------------------------------------------------------
db = Chroma(persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002'))

db.get(where={'Incident_ID': '20150417-0021'})