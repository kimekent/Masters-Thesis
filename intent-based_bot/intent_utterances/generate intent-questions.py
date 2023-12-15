# Token Management and File Operations
import os
from testing_functions import num_tokens_from_string, remove_history, save_file, open_file, adjust_similarity_scores_final_model_test

# OpenAI Libraries and functions
import openai
from testing_functions import gpt3_1106_completion, gpt3_0613_completion

# Libraries for initializing the retriever and the vector store
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import ast

# 1. Set up-------------------------------------------------------------------------------------------------------------
# Define directory paths
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\testing"
chroma_directory = path + r'\final_test\webhelp and websupport_vector_db'
prompt_logs_directory = path + r"\logs\answer_generation"
retriever_prompt_log_directory = path + r"\logs\retriever"

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = open_file(path + r"\openaiapikey.txt")
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2. Chatbot------------------------------------------------------------------------------------------------------------
def main(prompt):

    response = gpt3_1106_completion(prompt=prompt,
                                    model="gpt-4-1106-preview",
                                    log_directory=path + r"\logs\answer_generation",
                                    max_tokens=1000)
    return response

if __name__ == "__main__":
    main()