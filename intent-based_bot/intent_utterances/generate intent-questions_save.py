
import sys
from testing_functions import open_file
import os
import openai
import importlib

path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\testing"
# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = open_file(path + "\openaiapikey.txt")
openai.api_key = os.getenv("OPENAI_API_KEY")

sys.path.append(path + r"\final_test")
module = importlib.import_module("generate intent-questions")

import re
from testing_functions import open_file

data_string = open_file(r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\testing\final_test\intents.txt")

# Split the data into chunks for each intent
intents_data = data_string.split('intent: ')[1:]  # Skip the first empty split

# Initialize an empty dictionary to store intents and examples
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

for intent, utterances in d_intents.items():
    # Format the utterance examples for display
    formatted_utterances = '\n'.join([f"- {utterance}" for utterance in utterances])

    # Create the prompt with the current intent and its examples
    prompt = (f"Kannst du 10 neue Fragen generieren für den intent {intent}. "
              f"Hier sind ein paar Beispiele von Fragen, die zu diesem Intent passen:\n"
              f"{formatted_utterances}")

    # Call your module's main function with the prompt
    response = module.main(prompt)  # Replace 'module.main' with the actual function call you need
    print(response)
    responses_dict[intent].append(response)


