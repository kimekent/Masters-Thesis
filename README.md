# Master's Thesis
This repository contains the components for building, running and testing the intent-based Rasa chatbot and the intent-less chatbot discussed in my master's thesis. Bellow you will find a list of the files contained in this repository and instructions on how to run the scripts.

## Files and Content

#### Intent-Based Chatbot
- `data/`: Includes the intent-less Rasa NLU training data (`nlu.yml`), as well as the stories (`stories.yml`) and rules (`rules.yml`) that guide the dialogue management.
- `actions/`: Holds the custom action scripts for the chatbot.
- `config.yml`: Contains the NLU pipeline and policies.
- `domain.yml`: Outlines the chatbot's domain, detailing the intents, entities, slots, and action templates. It also includes the text responses the bot should use for each recognized intent.
- `credentials.yml`: Contains the necessary credentials for integrating the intent-less Rasa chatbot with Microsoft Teams.
- `endpoints.yml`: Specifies the endpoints for the chatbot backend services, such as the tracker store and action server.
- `models/`: Stores the final chatbot models. There are 5 models to choose from: 
	- 1: '20231216-180718-alternating-agreement.tar.gz'
	- 2: '20231216-163751-silent-shares.tar.gz'
	- 3: '20231216-182555-obtuse-parallel.tar.gz'
	- 4: '20231216-185922-chromatic-bevel.tar.gz'
	- BERT-Pipeline: '20231216-212421-impulsive-frequency.tar.gz'

Models 1-4 are trained on a spaCy pipeline with spaCy tokenizers and featurizers. The "BERT-Pipeline" model is trained on a pipeline usining BERT-embeddings ("distilbert-base-german-cased"), using the configuration specified in the `BERT_pipeline/`. Model 4 showed the best performance measured on F1 score.
- `BERT_pipeline/`: Contains the necessary files to set up the BERT pipeline. The BERT pipeline uses BERT embeddings for NLU.
- `test_data/`: Contains the test data used to test the model.
- `testing_results`: Stores the output from the testing phase of the chatbot. It includes detailed `intent_reports` for each iteration. Additionally, the latest iteration's results are captured in `intent_error.json`, `intent_confusion_matrix.png`, and `intent_histogram.png`.
- `MS_teams_files/`: Includes customized Rasa files essential for running the chatbot on Microsoft Teams.

#### Intent-Less Chatbot:
- `data/`: Contains the cleaned web support ticketing dataset and scraped web help articles.
- `ETL_web_help_articles/`: Contains scripts used to scrape and split web help articles.
- `create_chroma_db.py`: Script used to create the Chroma vector database.
- `webhelp_and_websupport_vector_db/`: Contains the Chroma vector database for storing and querying web support Q&A and web help articles.
- `functions.py`: Contains all functions required to run the intent-less chatbot.
- `chatbot.py`: Script to interact with the intent-less chatbot via the terminal. Requires adding an OpenAI API key and updating the file path.
- `prompts/`: Includes prompt `.txt` files for answer generation and question reformulation for the retriever.
- `gpt3_logs/`: Saves the logs from interactions with the intent-less chatbot.
- `openaiapikey.txt`: Placeholder text file. Paste the OpenAI API key here and update the path in `chatbot.py` to run the intent-less chatbot in terminal.
- `words_to_check_for_adjusted_similarity_score.txt`: List of words checked to rescore the retriever similarity scores.

#### Testing
- `testing_chatbot/`: Contains the scripts needed to run the testing chatbot, such as functions, logs, prompts.
- `testing_data/`: Includes all datasets used for training and testing the intent-less chatbot:
  - The cleaned web support ticketing dataset.
  - Test dataset comprising 80% of the ticketing dataset.
  - Conversation turn dataset to evaluate the memory component of the chatbot.
  - Test set for benchmarking the intent-based Rasa chatbot against the intent-less OpenAI chatbot.
  - Web help article collection.
  - Curated list of words for scaling similarity scores during retrieval.
- `testing_results/`: Stores the outcomes of the chatbot tests.
- `testing-websupport-and-webhelp-db/`: Chroma vector DB with the training web support Q&A and web help articles.
- `tests/`: Holds unit tests.
- `additional_german_stopwords.txt`: A text file listing extra German stopwords used in finding TF-IDF scores for each intent.
- `create_index.py`: A Python script that constructs an index, likely for quick retrieval of data or efficient database querying.
- `create_testing_dataset.py`: This script is responsible for creating the testing dataset.
- `openaiapikey.txt`: A placeholder storing the OpenAI API key.
- `testing_chroma_db_websupport-and-webhelp.py`: Script used to create the testing Chroma vector database.
- `TF-IDF_per_intent.py`: A Python script that calculates the Term Frequency-Inverse Document Frequency (TF-IDF) for each intent, a statistical measure used to evaluate the importance of a word to an intent in a collection or corpus.

#### Intent-Less Chatbot on Teams
- `bots/`: The core of the intent-less chatbot's functionality for Teams. It includes `echo_bot.py`, detailing the chatbot's operations, and contains all necessary components such as utility functions, prompt files, the vector database for queries, log files, the placeholder for the OpenAI API key, and state management files that track the chatbot's current context, query counts, and a text file with words used to adjust retrieval scores.
- `functions/`: Contains utility and helper functions used throughout the chatbot's codebase to perform various tasks and operations.
- `app.py`: The main Python executable script that initiates the chatbot application, serving as the entry point for the chatbot service.
- `config.py`: A configuration file in Python format which contains settings and variables that dictate how the chatbot operates within the Teams environment.

## Set up and Installation
To run the scripts the required dependencies can be installed with the following commands:

1. Clone a copy of this repo:
`git clone https://github.com/kimekent/Masters-Thesis.git`
2. Navigate to the root project folder.
`cd Masters-Thesis`
3. Install the dependencies  from the root level of the repo using: `pip install -r requirements.txt`


#### Intent-Based Chatbot
To run the intent-based Rasa chatbot:

1. Navigate to the intent-based_chatbot directory:
`cd intent-based_chatbot`
2. Install the necessary language model:
`python -m spacy download de_core_news_sm`
3. Start the Rasa shell in the terminal:
`rasa shell -m 'models\"model name"`
The list of models to choose from are listed under Files and Content -- Intent-based Chatbot.
4. In a separate terminal, start the actions server:
`cd intent-based_chatbot`
`rasa run actions`
5. Now, you can interact with the chatbot in the Rasa shell.


#### Intent-Less Chatbot
To run the intent-less chatbot:

1. Modify the root folder path in chatbot.py to your project's root folder.
2. Add your OpenAI API key to openaiapikey.txt located in root folder of this project.
3. You can run chatbot.py either from the terminal or run `\intent-less_chatbot\chatbot.pychatbot.py` in an IDE.
`python intent-less_chatbot/chatbot.py`

#### Testing
For testing:

1. Modify the root folder path in the .py files to your project's root folder.
2. Add your OpenAI API key to openaiapikey.txt located in root folder of this project.
3. Follow the instructions at the top of the test scripts to calculate metrics like BLEU-4, ROUGE-L, and BERTScore on saved generated answers datasets.

### Running Chatbots on Microsoft Teams
#### Prerequisites

1. Create an Azure Bot Service Instance.
2. Install [Ngrok](https://ngrok.com/download).

#### Intent-Based Chatbot
Integration with Microsoft Teams:

1. Replace `actions/actions.py` with the content from `MS_teams_files/teams_actions.py`.
2. Add `intent-based_chatbo/MS_teams_files/cards.py` to the `intent-based_chatbo/actions/` directory.
3. Update `intent-based_chatbot\domain.yml` with the content from `intent-based_chatbot/MS_teams_files/domain_teams.yml`.
4. Add Bot Framework credentials (app_id and app_password) in intent-based_chatbot/credentials.py. These credentials can be found in your Azure bot under 'Configurations' > 'Manage Password'.

#### Running the Intent-Based Chatbot on Teams
1. Navigate to intent-based_chatbot.
2. Start Rasa:
`rasa run`
3. In a separate terminal, start actions:
`rasa run actions`
4. Run Ngrok:
`ngrok http 5005`
5. In the Azure portal, set the messaging endpoint to your Ngrok URL followed by /webhooks/botframework/webhook.
6. Add Microsoft Teams as a channel and open the chatbot in Teams.

#### Running the Intent-Less Chatbot on Teams
1. Navigate to intent-less_chatbot `\intent-less_chatbot_on_teams`.
2. Run the chatbot:
`python app.py`
3. Start Ngrok on the same port:
`ngrok http [port]`
4. In the Azure portal, set the messaging endpoint to your Ngrok URL followed by /api/messages.
5. Add Microsoft Teams as a channel and open the chatbot in Teams.


### Suggested Utterances to Try
The chatbots are trained to answer websupport questions. Some sample utterances include:

#### Questions
- Wie lösche ich die englische Sprachversion einer Seite, ohne die Deutsche zu löschen?
- Wieso wird mein Text zweispaltig dargestellt?
- In der Kontaktbox wird die falsche Kontaktinformation dargestellt, wie kann ich diese anpassen.
- Wieso wird meine Infoveranstaltung in der Filterliste nicht dargestellt?
- Ich möchte eine Seite in die Sidenavigation aufnehmen, die keine direkte Unterseite ist. Geht das?
- wieso wird mein manueller teaser zweispaltig dargestellt?

#### To Test Fallback:
- Wie ist das Wetter heute?
- Wie kann ich einen Mailverlauf im CRM aufsetzten?
- Wie kann ich mich bei Google Analytics einloggen?

#### To Test Conversation Turn Capabilities of Intent-Less Chatbot
- Was ist ein Anker? Wie kann ich ihn einfügen?
- Wie lösche ich eine Seite? Es ist eine englische Seite?
- Wie kann ich eine Infobox löschen? Nun möchte ich sie wieder hinzufügen.

