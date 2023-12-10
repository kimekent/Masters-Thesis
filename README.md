# Masters-Thesis
This repository contains all the code used to create and test the intent-based rasa chatbot and the inten-less chatbot discussed in my masters thesis.

## Files and Content

### Intent-less Chatbot:

- `data/`: Contains the cleaned web support ticketing dataset and scraped web help articles.
- `gpt3_logs/`: Saves the logs from interactions with the intent-less chatbot.
- `prompts/`: Includes prompt .txt files for answer generation and question reformulation for the retriever.
- `webhelp_and_websupport_vector_db/`: Contains the Chroma vector database for storing and querying embeddings.
- `chatbot.py`: Script to interact with the intent-less chatbot via the terminal. Requires adding an API key and updating the file path.
- `create_chroma_db.py`: Used to create the Chroma vector database, storing web support Q&A and web help articles.
- `functions.py`: Contains all functions required to run the intent-less chatbot.
- `openaiapikey.txt`: Placeholder text file. Paste the OpenAI API key here and update the path in `chatbot.py`.
- `scraping_web_help_articles.py`: Script for scraping and storing web help articles.
- `split_web_help_articles.py`: Splits documents exceeding 3000 tokens into separate chunks.
- `words_to_check_for_adjusted_similarity_score.txt`: Lists words checked by the `adjust_similarity_scores` function.


Intent-based chatbot
- All files needed to run the rasa chatbot, plus the test
