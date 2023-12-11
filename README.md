# Masters-Thesis
This repository contains all the code used to create and test the intent-based rasa chatbot and the inten-less chatbot discussed in my masters thesis.

This repository contains the components for building and running an intent-based chatbot using the Rasa framework. Below is a description of the directories and files included in this project.

## Files and Content

### Intent-based chatbot
- `data/`: Includes the intent-less Rasa NLU training data (`nlu.yml`), as well as the stories (`stories.yml`) and rules (`rules.yml`) that guide the dialogue management.
- `actions/`: This directory holds the custom action scripts for the chatbot.
- `config.yml`: This file contains the NLU pipeline and policies.
- `domain.yml`: Outlines the chatbot's domain, detailing the intents, entities, slots, and action templates. It also includes the text responses the bot should use for each                  recognized intent.
- `credentials.yml`: Contains the necessary credentials for integrating the intent-less rasa chat bot with Microsoft Teams.
- `endpoints.yml`: Specifies the endpoints for the chatbot backend services, such as the tracker store and action server.
- `models/`: Stores the final chatbot models. There are two versions: one trained with spaCy embeddings and another trained with BERT embeddings using the configuration                   specified in the `BERT_pipeline/`.
- `BERT_pipeline/`: Contains the necessary files to set up the BERT pipeline. The BERT pipeline uses BERT embeddings for NLU.
- `test_data/`: Contains the test data used to test the model
- `testing_results`: Stores the output from the testing phase of the chatbot. It includes detailed `intent_reports` for each iteration. Additionally, the latest iteration's                       results are captured in `intent_error.json`, `intent_confusion_matrix.png`, and `intent_histogram.png.
- `MS_teams_files/`: This directory includes customized rasa files that are essential for running the chatbot on Microsoft Teams.

### Intent-less Chatbot:

- `data/`: Contains the cleaned web support ticketing dataset and scraped web help articles.
- `ETL_web_help_articles/`: Contains scripts used to scrape and split web help articles.
- `create_chroma_db.py`: Script used to create the Chroma vector database.
- `webhelp_and_websupport_vector_db/`: Contains the Chroma vector database for storing and querying web support Q&A and web help articles.
- `functions.py`: Contains all functions required to run the intent-less chatbot.
- `chatbot.py`: Script to interact with the intent-less chatbot via the terminal. Requires adding an OpenAI API key and     updating the file path.
- `prompts/`: Includes prompt .txt files for answer generation and question reformulation for the retriever.
- `gpt3_logs/`: Saves the logs from interactions with the intent-less chatbot.
- `openaiapikey.txt`: Placeholder text file. Paste the OpenAI API key here and update the path in `chatbot.py`.
- `words_to_check_for_adjusted_similarity_score.txt`: List of words checked to rescore the retreiver similarity scores.

### testing
- `testing_chatbot/`: This folder contains the scripts needed to run the testing chatbot, such as functions, logs, prompts
- `testing_data/`: Includes all datasets used for training and testing the intent-less chatbot:
  - The cleaned web support ticketing dataset.
  - Test dataset comprising 80% of the ticketing dataset.
  - Conversation turn dataset to evaluate the memory component of the chatbot.
  - Test set for benchmarking the intent-based Rasa chatbot against the intent-less OpenAI chatbot.
  - Web help article collection.
  - Curated list of words for adjusted similarity scoring in retrieval.
- `testing_results/`: Stores the outcomes of the chatbot tests
- `testing-websupport-and-webhelp-db/`: Chroma vector db with the training  web support Q&A and web help articles.
- `tests/`: Holds unit tests
- `additional_german_stopwords.txt`: A text file listing extra German stopwords used in finding TF-IDF scores of each intent.
- `create_index.py`: A Python script that constructs an index, likely for quick retrieval of data or efficient database querying.
- `create_testing_dataset.py`: This script is responsible for creating the testing dataset 
- `openaiapikey.txt`: A placeholder storing the OpenAI API key.
- `testing_chroma_db_websupport-and-webhelp.py`: Script used to create testing Chroma vector database
- `TF-IDF_per_intent.py`: A Python script that calculates the Term Frequency-Inverse Document Frequency (TF-IDF) for each intent, which is a statistical measure used to evaluate the importance of a word to an intent in a collection or corpus.
  

#### Additional Information

  - To integrate with Microsoft Teams, you will need to:
    1. Replace the `actions/actions.py` file with the contents from `MS_teams_files/teams_actions.py`.
    2. Add the `MS_teams_files/cards.py` file to the `actions/` directory.
    3. Replace the content in the `domain.yml` file with the content from `MS_teams_files/domain_teams.yml`.

#### Instructions for Microsoft Teams Integration

1. **Updating Actions**: Replace the existing `actions.py` script with `teams_actions.py` to ensure the chatbot actions are compatible with Microsoft Teams.

2. **Adding Card Functionality**: Transfer `cards.py` into the `actions/` directory to enable rich message formats, such as adaptive cards in Microsoft Teams.

3. **Domain Modification**: Update the `domain.yml` with `domain_teams.yml` to align the chatbot's responses and actions with the conversational flows and capabilities supported by Microsoft Teams.
## Setup and Installation

[Provide instructions here on how to set up and install the chatbot, including any prerequisites.]

## Usage

[Provide instructions here on how to train the bot, run it, and interact with it.]

## Contributing

[If open to contributions, include instructions on how contributors can help with your project.]

## License

[State the license under which this project is available.]

Remember to replace placeholder text with specific information about how to set up, configure, and use your chatbot. Include any additional instructions or explanations as necessary.

