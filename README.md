# Masters-Thesis
This repository contains the components for building, running and testing the intent-based rasa chatbot and the intent-less chatbot discussed in my masters thesis. bellow you will find a list of the files contained in this repository and instructions on how to run the scripts.

## Files and Content

#### Intent-based Chatbot
- `data/`: Includes the intent-less Rasa NLU training data (`nlu.yml`), as well as the stories (`stories.yml`) and rules (`rules.yml`) that guide the dialogue management.
- `actions/`: Holds the custom action scripts for the chatbot.
- `config.yml`: Contains the NLU pipeline and policies.
- `domain.yml`: Outlines the chatbot's domain, detailing the intents, entities, slots, and action templates. It also includes the text responses the bot should use for each recognized intent.
- `credentials.yml`: Contains the necessary credentials for integrating the intent-less Rasa chatbot with Microsoft Teams.
- `endpoints.yml`: Specifies the endpoints for the chatbot backend services, such as the tracker store and action server.
- `models/`: Stores the final chatbot models. There are two versions: one trained with spaCy embeddings and another trained with BERT embeddings using the configuration specified in the `BERT_pipeline/`.
- `BERT_pipeline/`: Contains the necessary files to set up the BERT pipeline. The BERT pipeline uses BERT embeddings for NLU.
- `test_data/`: Contains the test data used to test the model.
- `testing_results`: Stores the output from the testing phase of the chatbot. It includes detailed `intent_reports` for each iteration. Additionally, the latest iteration's results are captured in `intent_error.json`, `intent_confusion_matrix.png`, and `intent_histogram.png`.
- `MS_teams_files/`: Includes customized Rasa files essential for running the chatbot on Microsoft Teams.

#### Intent-less Chatbot:
- `data/`: Contains the cleaned web support ticketing dataset and scraped web help articles.
- `ETL_web_help_articles/`: Contains scripts used to scrape and split web help articles.
- `create_chroma_db.py`: Script used to create the Chroma vector database.
- `webhelp_and_websupport_vector_db/`: Contains the Chroma vector database for storing and querying web support Q&A and web help articles.
- `functions.py`: Contains all functions required to run the intent-less chatbot.
- `chatbot.py`: Script to interact with the intent-less chatbot via the terminal. Requires adding an OpenAI API key and updating the file path.
- `prompts/`: Includes prompt `.txt` files for answer generation and question reformulation for the retriever.
- `gpt3_logs/`: Saves the logs from interactions with the intent-less chatbot.
- `openaiapikey.txt`: Placeholder text file. Paste the OpenAI API key here and update the path in `chatbot.py`.
- `words_to_check_for_adjusted_similarity_score.txt`: List of words checked to rescore the retriever similarity scores.

#### Testing
- `testing_chatbot/`: Contains the scripts needed to run the testing chatbot, such as functions, logs, prompts.
- `testing_data/`: Includes all datasets used for training and testing the intent-less chatbot:
  - The cleaned web support ticketing dataset.
  - Test dataset comprising 80% of the ticketing dataset.
  - Conversation turn dataset to evaluate the memory component of the chatbot.
  - Test set for benchmarking the intent-based Rasa chatbot against the intent-less OpenAI chatbot.
  - Web help article collection.
  - Curated list of words for adjusted similarity scoring in retrieval.
- `testing_results/`: Stores the outcomes of the chatbot tests.
- `testing-websupport-and-webhelp-db/`: Chroma vector DB with the training web support Q&A and web help articles.
- `tests/`: Holds unit tests.
- `additional_german_stopwords.txt`: A text file listing extra German stopwords used in finding TF-IDF scores for each intent.
- `create_index.py`: A Python script that constructs an index, likely for quick retrieval of data or efficient database querying.
- `create_testing_dataset.py`: This script is responsible for creating the testing dataset.
- `openaiapikey.txt`: A placeholder storing the OpenAI API key.
- `testing_chroma_db_websupport-and-webhelp.py`: Script used to create the testing Chroma vector database.
- `TF-IDF_per_intent.py`: A Python script that calculates the Term Frequency-Inverse Document Frequency (TF-IDF) for each intent, a statistical measure used to evaluate the importance of a word to an intent in a collection or corpus.

#### Intent-less Chatbot on Teams
- `bots/`: The core of the intent-less chatbot's functionality for Teams. It includes `echo_bot.py`, detailing the chatbot's operations, and contains all necessary components such as utility functions, prompt files, the vector database for queries, log files, the placeholder for the OpenAI API key, and state management files that track the chatbot's current context, query counts, and a text file with words used to adjust retrieval scores.
- `functions/`: Contains utility and helper functions used throughout the chatbot's codebase to perform various tasks and operations.
- `app.py`: The main Python executable script that initiates the chatbot application, serving as the entry point for the chatbot service.
- `config.py`: A configuration file in Python format which contains settings and variables that dictate how the chatbot operates within the Teams environment.
- `requirements.txt`: Lists all the Python dependencies required to run the chatbot.

These components collectively make up the chatbot application for Teams. Ensure you have all dependencies installed as listed in `requirements.txt` before running `app.py`.


## Instructions
To run any of these scripts, first install the requirements list by cd into root project folder clone the directory with then with pip install requirements.txt

#### Intent-based chatbot
To run the intent-based rasa chatbot on terminal navigate to \intent-based_chatbot
before running chatbot execute python -m spacy download de_core_news_sm
then run command rasa shell
in seperate terminal start actions server with rasa run acitons
talk to chatbot

#### Intent-less chatbot
To run intent-less chatbot in terminal navigate to \intent-less_chatbot\chatbot.py or run the \intent-less_chatbot\chatbot.py in an IDE.
Before running the script change the root folder path to your root folder and add your OpenAI API key to \openaiapikey.txt

#### Testing
To run the tests make sure to change the root folder path to your root folder and add your OpenAI API key to \openaiapikey.txt
The tests such as calculating BLEU-4, ROUGE-L, and BERTScore can also be calculated on the saved genereated datasets. At the top of script there is a
description which tells you which steps to execute if you wish to do so.

#### Run chatbots on Microsoft Teams
Prerequisites
To run the bots on MS Teams a Azure Bot Service Instance need to be created. 
Install Ngrok

intent-based chatbot
 - To integrate with Microsoft Teams, you will need to:
    1. Replace the `actions/actions.py` file with the contents from `MS_teams_files/teams_actions.py`.
    2. Add the `MS_teams_files/cards.py` file to the `actions/` directory.
    3. Replace the content in the `domain.yml` file with the content from `MS_teams_files/domain_teams.yml`.
	4. Add botframework credentials (app_id and app_password) to run chatbot with Microsoft Teams frontend \intent-based_chatbot\credentials.py
These credentials can be found in your azure bot under 'congigurations' > Manage Password
1. **Updating Actions**: Replace the existing `actions.py` script with `teams_actions.py` to ensure the chatbot actions are compatible with Microsoft Teams.

2. **Adding Card Functionality**: Transfer `cards.py` into the `actions/` directory to enable rich message formats, such as adaptive cards in Microsoft Teams.

3. **Domain Modification**: Update the `domain.yml` with `domain_teams.yml` to align the chatbot's responses and actions with the conversational flows and capabilities supported by Microsoft Teams.
## Setup and Installation

Running the chatbot
Navigate to \intent-bases_chatbot
In terminal run rasa run
In seperate terminal run rasa run acitons
In separate terminal run ngrok http 5005
In Azure Bot service instance add the ngrok https url  Azure portal, open the resource page of your bot channel registration and navigate to „Settings > Configuration“. As messaging endpoint, enter your individual ngrok URL followed by „/webhooks/botframework/webhook“.
On ressource page on bot channel navigate to channels "add Chanel" Microsoft teams and then Click open in Microsoft Teams
YOu can now chat to chatbot through teams


Intent-less Chatbot
Navigate to \intent-less chatbot 
In terminal run python app.py
In seperate terminalrun ngrok http [protal where app.py is running on]
In Azure Bot service instance add the ngrok https url  Azure portal, open the resource page of your bot channel registration and navigate to „Settings > Configuration“. As messaging endpoint, enter your individual ngrok URL followed '/api/messages'
On ressource page on bot channel navigate to channels "add Chanel" Microsoft teams and then Click open in Microsoft Teams
YOu can now chat to chatbot through teams


### To 


#### Additional Information

 

#### Instructions for Microsoft Teams Integration



[Provide instructions here on how to set up and install the chatbot, including any prerequisites.]

## Usage

[Provide instructions here on how to train the bot, run it, and interact with it.]

## Contributing

[If open to contributions, include instructions on how contributors can help with your project.]

## License

[State the license under which this project is available.]

Remember to replace placeholder text with specific information about how to set up, configure, and use your chatbot. Include any additional instructions or explanations as necessary.

