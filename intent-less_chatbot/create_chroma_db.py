"""
This file was used to create the Chroma vector database. It includes the embeddings and metadata of
all web support Q&A and web help articles.
"""
# Import libraries
import os
import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from functions import open_file, replace_links_with_placeholder

# Set file paths
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Github\Intent-less Chatbot" # change
os.environ["OPENAI_API_KEY"] = open_file(path + r"\openaiapikey.txt") # change
# The persist_directory is where the embeddings are stored on disk
persist_directory = path + r"\webhelp_and_websupport_vector_db"
# Path to CSV file containing the web support question and answers
websupport_questions = path + r"\data\cleaned_websupport_questions_with_intents_utf-8.csv"

# Embeddings that are used for the Chroma Database
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Read questions and answers from the CSV file
df = pd.read_csv(websupport_questions, encoding="utf-8")
filtered_df = df[df["Anfrageart"] == "Dokumentation nicht gelesen"]
questions = filtered_df["Beschreibung"].tolist()
questions = [replace_links_with_placeholder(i) for i in questions]
incident_ids = filtered_df["Incident-ID"].tolist()
answers = filtered_df["Lösungsbeschreibung"].tolist()
intents = filtered_df["intent"].tolist()

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
            "Source": "websupport question",
            "Answer": answer,
            # This corresponds to the ID of the question from the web support dataset
            "Incident_ID": incident_id,
            "intent": intent,
            "id": i  # Add an "id" field with the current index i as its value
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
i = len(questions) + 1 # Initialize the "i" variable outside the loop
for filename in os.listdir(path + "\data\chunked_webhelp_articles"):
    if filename.endswith(".txt"):
        with open(os.path.join(path + "\data\chunked_webhelp_articles", filename), "r", encoding="utf-8") as file:
            content = file.read()
            document = Document(
                page_content=content,
                metadata={
                    "Source": "webhelp-article",
                    "Key": content.split("Key: ")[1].split("\n")[0].strip(),
                    "Link": content.split("Link: ")[1].strip(),
                    "id": i  # Set the "id" to the value of "i"
                }
            )
            docs.append(document)
            i += 1  # Increment "i" for the next document

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)


# Check to see if database exists and was configured properly
db = Chroma(persist_directory=persist_directory,
                                 embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002'))

db.get(where={"Incident_ID": "20150417-0021"})