"""
This script was used to generate additional utterances of the intents, to train
the intent-based chatbot. The intent are generated using OpenAIs 'gpt-4-1106-preview' model.

To run this script set the 'path' variable to the root directory of this project and add
the OpenAI API key to openaiapikey.txt, located in the root directory of this project.
"""


# 1. Set up-------------------------------------------------------------------------------------------------------------
# Set path variable and OpenAI API key
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis"
# Set the API key as an environment variable
from testing_chatbot.testing_functions import open_file
import openai
import os
os.environ["OPENAI_API_KEY"] = open_file(path + "\openaiapikey.txt")
openai.api_key = os.getenv("OPENAI_API_KEY") # Add OpenAI API key to this .txt file

# Import packages
import sys
from testing_functions import open_file
import importlib
import re

# 2. Generate Utterances------------------------------------------------------------------------------------------------
# Function to generate utterance
def generate_utterance(prompt):

    response = gpt3_1106_completion(prompt=prompt,
                                    model="gpt-4-1106-preview",
                                    log_directory=path + r"\intent-based_bot\creating_additonal_intent_utterances\gpt-4_log",
                                    max_tokens=1000)
    return response

# Create dictionary with intent as key and sample utterances as values
data_string = open_file(path + r"\intent-based_bot\creating_additonal_intent_utterances")

# Split the data into chunks for each intent
intents_data = data_string.split('intent: ')[1:]  # Skip the first empty split

# Initialize an empty dictionary to store intents and sample utterances
d_intents = {}

for intent_data in intents_data:
    # Split the intent data into intent and examples
    lines = intent_data.split('\n')
    intent = lines[0].strip()
    examples = [line.strip('- ').strip() for line in lines if line.startswith('    - ')]

    # Add to the dictionary
    d_intents[intent] = examples

# Print the result
print(d_intents)

responses_dict = {intent: [] for intent in d_intents.keys()}

# Generate new utterances per intent
for intent, utterances in d_intents.items():
    # Format the utterance examples for display
    formatted_utterances = '\n'.join([f"- {utterance}" for utterance in utterances])

    # Create the prompt with the current intent and its examples
    prompt = (f"Kannst du 10 neue Fragen generieren für den intent {intent}. "
              f"Hier sind ein paar Beispiele von Fragen, die zu diesem Intent passen:\n"
              f"{formatted_utterances}. Verwende unterschiedliche Satzstrukturen und Begriffe.")

    response = generate_utterance(prompt)
    print(response)
    responses_dict[intent].append(response)


