"""
In this script the latest version of the intentless chatbot is run on questions that pertain to one of the intents
that the intent-based chatbot was trained on. Per intent ten questions are passed to the chatbot.
The same questions will also be fed to the intent-based chatbot, allowing to compare the performance in terms of accuracy
between the two bots.
"""

# 1. Set up-------------------------------------------------------------------------------------------------------------

# Standard Libraries
import os
import pandas as pd

# Libraries to run chatbot
import openai
from testing_functions import open_file
from time import sleep

# Libraries to import chatbotL
import importlib
import sys

# Define variables and paths
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\testing"
# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = open_file(path + "\openaiapikey.txt")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import testing chatbot
sys.path.append(path + r"\final_test")
module = importlib.import_module("rasa_openai_test")

# Import questions
questions_df = pd.read_csv(path + r"\testing_data\rasa_openai_comparison.csv")
questions = questions_df["question"].tolist()

# 2. Run chatbot--------------------------------------------------------------------------------------------------------
l_generated_response = [] # Initialize list to store the generated responses
for question in questions:
    response = module.main(question)
    l_generated_response.append(response)
    sleep(30)

# 3. Save the generated responses in a csv file-------------------------------------------------------------------------
questions_df["generated_answers"] = l_generated_response
questions_df.to_csv(path + r"\testing_results\final_test.csv")