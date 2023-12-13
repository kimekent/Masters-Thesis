"""
This script was used to calculate the token count for each web help article.
Articles exceeding 3000 tokens were then manually split. By manually splitting the files,
the chunks were guaranteed to be  complete and make sense on their own.

To run this script update the 'path' variable to the root directory of this project.
"""


# 1. Set up-------------------------------------------------------------------------------------------------------------
# Set path to root directory and intent-less chatbot folder
path = r'C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis' # Change
intent_less_path = path + r'\intent-less_chatbot'

# Import packages
import os

# Initialize an empty dictionary to store the results
file_lengths = {}

# 2. Count tokens per web help article----------------------------------------------------------------------------------
# Iterate through the files in the directory
for filename in os.listdir(intent_less_path + r'\data\scraped_web_support_files'):
    if filename.endswith('.txt'):
        # Get the full file path
        file_path = os.path.join(intent_less_path + r'\data\scraped_web_support_files', filename)

        # Open the file and read its content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Calculate the character length of the content
        char_length = len(content)

        # Add the filename and character length to the dictionary
        file_lengths[filename] = char_length

# Print the dictionary
for filename, length in file_lengths.items():
    print(f'{filename}: {length} characters')

# 3. Get name of txt files that have more than 3000 characts------------------------------------------------------------
# Set the character length threshold
threshold = 3000

# Iterate through the dictionary and print filenames that exceed the threshold
for filename, char_length in file_lengths.items():
    if char_length > threshold:
        print(f'{filename}: {char_length} characters')