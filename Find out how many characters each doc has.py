import os
# Create dictionary with filename and numer of characters in txt file
# Specify the directory where the .txt files are located
directory = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\OpenAI Chatbot\data"

# Initialize an empty dictionary to store the results
file_lengths = {}

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        # Get the full file path
        file_path = os.path.join(directory, filename)

        # Open the file and read its content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Calculate the character length of the content
        char_length = len(content)

        # Add the filename and character length to the dictionary
        file_lengths[filename] = char_length

# Print the dictionary
for filename, length in file_lengths.items():
    print(f"{filename}: {length} characters")

#Get name of txt files that have more than 3000 characters_________________
# Set the character length threshold
threshold = 3000

# Iterate through the dictionary and print filenames that exceed the threshold
for filename, char_length in file_lengths.items():
    if char_length > threshold:
        print(f"{filename}: {char_length} characters")

# Add txt files together
# Specify the directory where the text documents are located

# Output file where concatenated content will be saved
output_file = directory + r"input.txt"

# Empty string to store concatenated content
concatenated_content = ""

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        # Get the full file path
        file_path = os.path.join(directory, filename)

        # Read the file's content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Append the content to the concatenated string with "-- new document--" separator
        concatenated_content += content + "\n-- new document--\n"

# Write the concatenated content to the output file
with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.write(concatenated_content)

