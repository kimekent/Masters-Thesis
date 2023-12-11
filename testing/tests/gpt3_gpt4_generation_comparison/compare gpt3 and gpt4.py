"""
This script is designed to compare the performance of GPT-3.5 and GPT-4, by testing them with a set of websupport
questions. This comparison aims to evaluate and understand the differences in response quality and content between
GPT-3.5 and GPT-4.
"""
# 1. Set up-------------------------------------------------------------------------------------------------------------

# Standard Libraries
import os
import pandas as pd
import sys

# Libraries to run chatbot
import openai
from testing_functions import open_file
import importlib
from time import sleep

# Define variables and paths
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\testing"
# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = open_file(path + "\openaiapikey.txt")
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2. Run chatbot--------------------------------------------------------------------------------------------------------

# Import testing chatbot
sys.path.append(path + r"\gpt3_gpt4_generation_comparison")
script_name = "testing_chatbot_gpt4_gpt3"  # Name of testing chatbot script
module = importlib.import_module(script_name)

# Import questions
questions_df = pd.read_csv(path + r"\testing_data\test_dataset.csv")
questions = questions_df["Beschreibung"].tolist()

# Initialize lists to store generated answers
l_generated_response_gpt3 = []
l_generated_response_gpt4 = []

# Text generation models that are tested
models = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"]

for model in models:
    for question in questions:
        response = module.main(question, model)

        if model == "gpt-3.5-turbo-1106":
            l_generated_response_gpt3.append(response)
        elif model == "gpt-4-1106-preview":
            l_generated_response_gpt4.append(response)

        sleep(5)

# 3. Save generated answers to csv file---------------------------------------------------------------------------------

questions_df["generated_answers_gpt3"] = l_generated_response_gpt3
questions_df["generated_answers_gpt4"] = l_generated_response_gpt4

questions_df.to_csv(path + r"\testing_results\gpt3_and_gpt4_comparison.csv")
