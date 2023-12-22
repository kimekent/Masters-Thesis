"""
This script contains the code to run the final version of the intent-less chatbot.
It prints the original question, its reformulated version, the current prompt,
and the chatbot's response to the console.
Note that the print statement for the current prompt, which can be long, can be disabled by commenting it out
(see line 133).

To run this script update the 'path' variable to the root directory of this project and add your OpenAI API key to
'openaiapikey.txt' in the root directory of this project.
"""


# 1. Set up-------------------------------------------------------------------------------------------------------------
# Set path to project directory and define OpenAI API key
import sys
path = r'C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\Masters-Thesis'  # Change
intent_less_path = path + r'\intent-less_chatbot'
sys.path.append(intent_less_path)

from functions import open_file
import openai
import os
os.environ['OPENAI_API_KEY'] = open_file(path + '\openaiapikey.txt')
openai.api_key = os.getenv('OPENAI_API_KEY')  # Add OpenAI API key to this .txt file

# Import packages and functions
# Token Management and File Operations
from functions import num_tokens_from_string, remove_history, save_file, adjust_similarity_scores

# OpenAI Libraries and functions
from functions import gpt_completion

# Libraries for initializing the retriever and the Chroma vector store
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import ast

# Library and function for sending mails
import win32com.client as win32
from functions import send_email

# Define additional paths
chroma_directory = intent_less_path + r'\webhelp_and_websupport_vector_db'
prompt_logs_directory = intent_less_path + r'\gpt3_logs\prompt'
retriever_prompt_log_directory = intent_less_path + r'\gpt3_logs\retriever_prompt'

# Initialize Chroma vector database with text-embedding-ada-002
vectordb_websupport_bot = Chroma(persist_directory=chroma_directory,
                                 embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002'))

# Import list of words to check for re-scoring of retriever similarity scores.
words_to_check = ast.literal_eval(open_file(intent_less_path + r'/words_to_check_for_adjusted_similarity_score.txt'))


# 2. Chatbot------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Load prompts
    # This prompts is used to restructure question for the retriever. For each new session the history is deleted.
    retriever_prompt = remove_history(open_file(intent_less_path + r'\prompts\retriever_prompt.txt'),
                                      '<<newest message>>(.*?)<<oldest message>>',
                                      '<<newest message>>\n<<history>>\n<<oldest message>>')

    # This prompt is used to answer to question. For each new session the history is deleted.
    question_prompt = remove_history(open_file(intent_less_path + r'\prompts\prompt_answer.txt'),
                                     'HISTORY:(.*?)<<history>>',
                                     'HISTORY: <<history>>')

    query_count = 0  # Load trackers, Initialize query count

    l_retriever_history = []  # This list will contain the previous questions and answers.
    l_qa_history = []  # This list will contain the previous questions and answers for the Q&A prompt.

    while True:
        query = input('Enter your question here: ')  # Prompt the user for a question
        print('query: ' + query)
        query_count += 1

        if query_count > 1:
            # Reformulate the current question to include context from previous turns for better document retrieval
            # in multi-question sessions.
            current_retriever_prompt = retriever_prompt.replace('<<query>>', query)
            print("current retreiver prompt: " + current_retriever_prompt)
            restructured_query = gpt_completion(prompt=current_retriever_prompt,
                                                model='gpt-4-1106-preview',
                                                log_directory=retriever_prompt_log_directory)
            print('restructured query: ' + restructured_query)
        else:
            restructured_query = query

        # Initialize lists per question.
        l_webhelp_articles = []  # This list will contain all retrieved webhelp articles.
        l_webhelp_questions = []  # This list will contain all retrieved websupport questions.

        # Count how many tokens are in the prompt. In the beginning the token count only includes the initial prompt.
        total_tokens = num_tokens_from_string(question_prompt, encoding='cl100k_base')

        # Perform retrieval based on the user's query
        results = vectordb_websupport_bot.similarity_search_with_score(restructured_query, k=60)

        first_document_score = results[0][1]  # Extract the score of the first document

        # If cosine distance is below 0.3 continue answering question. Else hand off to web support representative.
        if first_document_score < 0.3:
            results = adjust_similarity_scores(results, question=restructured_query, word_intent_dict=words_to_check, multiplier=0.8)
            for doc in results:
                doc = doc[0]
                if doc.metadata.get('Source') == 'webhelp-article':
                    link = doc.metadata.get('Link')
                    webhelp_article_content = doc.page_content
                    context = f' {webhelp_article_content}\nLink: {link}'
                    l_webhelp_articles.append(context)

                elif doc.metadata.get('Source') == 'websupport question':
                    websupport_question = doc.page_content
                    websupport_answer = doc.metadata.get('Answer', 'No answer found')
                    # Format the question and answer
                    context = f'Q: {websupport_question}\nA: {websupport_answer}'
                    l_webhelp_questions.append(context)

                # Get the number of tokens
                tokens = num_tokens_from_string(context, encoding='cl100k_base')

                # If adding the answer would exceed the token limit, break out of the loop.
                if total_tokens + tokens > 14000:
                    break
                else:
                    total_tokens += tokens

            # Construct a prompt for GPT-3.5 Turbo based on the user's question
            current_prompt = question_prompt \
                .replace('<<query>>', restructured_query) \
                .replace('<<websupport_questions>>', '\n'.join(l_webhelp_questions)) \
                .replace('<<webhelp_article>>', ' '.join(l_webhelp_articles))
            #print('current prompt: ' + current_prompt)

            # Generate answer to prompt
            response = gpt_completion(prompt=current_prompt, log_directory=intent_less_path + r'\gpt3_logs\prompt',
                                      model='gpt-4-1106-preview', max_tokens=1000)
            print('response: ' + response)


        else:
            print("Leider kann ich deine Frage nicht beantworten. "
                  "Soll ich eine Websupport-Ticket mit deiner Frage eröffnen? (Antworte mit 'Ja' oder 'Nein')")

            yes_inputs = ['ja', 'yes', 'jawohl', 'yep', 'j', 'ok']
            no_inputs = ['nein', 'ne', 'no', 'nö', 'nope', 'n']

            user_input = input().strip().lower()

            if any(choice in user_input for choice in yes_inputs):
                websupport_ticket = restructured_query

                try:
                    print('Ich habe ein Mail in Outlook erstellt mit deiner Anfrage.')
                    outlook = win32.Dispatch('Outlook.Application')  # Starts Outlook application
                    new_email = outlook.CreateItem(0)  # Creates new email item
                    new_email.To = 'mefabe7562@marksia.com'  # for testing purposes a temp. email address is used
                    new_email.Body = restructured_query  # body of email
                    # new_email.Subject =
                    new_email.Display(True)  # Displays the new email item
                except:
                    while True:

                        print('Das ist deine Nachricht: ' + websupport_ticket)
                        print("Möchtest du die Nachricht noch bearbeiten, bevor ein Ticket daraus erstellt "
                              "wird? (Antworte mit 'Ja' oder 'Nein')")

                        edit_input = input().strip().lower()

                        if any(choice in edit_input for choice in yes_inputs):
                            print('Bitte gib die bearbeitete Nachricht ein:')
                            websupport_ticket = input()
                        elif any(choice in edit_input for choice in no_inputs):
                            send_email(restructured_query)
                            print('Email sent successfully!')
                            break
                        else:
                            print("Ungültige Eingabe. Bitte antworte mit 'Ja' oder 'Nein'.")
            if any(choice in user_input for choice in no_inputs):
                print("Falls noch nicht nachgeschaut, könnten dir die (Webhelp)[https://applsupport.hslu.ch/webhelp/]"
                      "behilflich sein. Ansonsten erreichst du den Websupport unter websupport@hslu.ch[mailto:websupport@hslu.ch]")
            else:
                print("Ungültige Eingabe. Bitte antworte mit 'Ja' oder 'Nein'.")

        # Add memory to retriever
        # Count how many tokens the retriever prompt has
        tokens_retriever = num_tokens_from_string(retriever_prompt, encoding='cl100k_base')
        l_retriever_history.append("human: " + restructured_query + '\n' + "AI: " + response)
        if tokens_retriever > 3000:  # If token limit is reached, delete latest conversation turn
            l_retriever_history = l_retriever_history[1:]

        # Reverse list, so newest chat history is added to the top of prompt
        l_reversed_retriever_history = l_retriever_history[::-1]

        # Delete old history and add new history with the latest Q&A to prompt
        retriever_prompt = remove_history(retriever_prompt,
                                          '<<newest message>>(.*?)<<oldest message>>',
                                          '<<newest message>>\n<<history>>\n<<oldest message>>')

        retriever_prompt = retriever_prompt.replace('<<history>>',
                                                    '\n'.join(l_reversed_retriever_history))
        print("retriever promt with history" + retriever_prompt)

        # Save history to retriever prompt.
        save_file(retriever_prompt, intent_less_path + r'\prompts\retriever_prompt.txt')

        # Add memory to Q&A prompt
        # In order not to exceed token length, only the last two conversation turns are added as history
        # to question_prompt.

        # Get last two conversation turns
        if len(l_qa_history) > 3:
            l_qa_history = l_qa_history[1:]

        last_QA_pair = f'Human: {restructured_query} \nAI: {response}\n'
        l_qa_history.append(last_QA_pair)

        # Delete all conversation turns
        question_prompt = remove_history(question_prompt,
                                         'Human:(.*?)<<history>>',
                                         '<<history>>')
        # Add new conversation turns
        question_prompt = question_prompt.replace('<<history>>', ' '.join(l_qa_history) + '\n<<history>>')