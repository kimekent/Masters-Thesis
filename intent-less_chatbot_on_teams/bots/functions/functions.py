from langchain.memory import ConversationBufferWindowMemory,ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate

def prompt_memory(memory_type, llm, max_conversation_turns=None, max_token_limit=None, memory_file_path=None):
    if memory_type == 'ConversationSummaryBufferMemory':
        with open(memory_file_path, 'r', encoding='utf-8') as file:
            memory_prompt_text = file.read()
            memory_prompt = PromptTemplate(input_variables=['summary', 'new_lines'], template=memory_prompt_text)
            memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_token_limit, prompt=memory_prompt)

    elif memory_type == 'ConversationBufferWindowMemory':
        memory = ConversationBufferWindowMemory(k=max_conversation_turns)

    elif memory_type == 'ConversationBufferMemory':
        memory = ConversationBufferMemory()

    else:
        raise ValueError(f"Invalid memory type: {memory_type}")

    return memory


from time import time, sleep
import openai
import backoff
import requests

@backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_time=60)
def gpt3_1106_completion(prompt, model='gpt-3.5-turbo-1106', temperature=0.7, max_tokens=150, log_directory=None):
    """
    Generate a completion for a given prompt using OpenAI's GPT-3.5-turbo chat model.

    Parameters:
    prompt (str): The prompt to send to GPT-3.5-turbo.
    model (str): The model to use. Default is 'gpt-3.5-turbo'.
    temperature (float): The temperature to use for the completion. Default is 0.7.
    max_tokens (int): The maximum number of tokens to generate. Default is 150.

    Returns:
    str: The generated completion text.
    """
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            try:
                # Extract and save the response
                text = response.choices[0].message.content.strip()
                filename = f'{time()}_gpt3.txt'
                with open(f'{log_directory}/{filename}', 'w') as outfile:
                    outfile.write('PROMPT:\n\n' + prompt + '\n\n====== \n\nRESPONSE:\n\n' + text)
                return text
            except:
                print("error saving to log")
        except requests.exceptions.RequestException as e:
            return "Es tut uns leid, es scheint, dass der Server derzeit überlastet ist. Versuche es später nocheinmal." \
                   "In der Zwischenzeit hilft dir vielleicht die Webhelp weiter: https://applsupport.hslu.ch/webhelp/" \
                   "In dringenden Fällen kannst du dich gerne beim Websupport melden: [mailto:websupport@hslu.ch]"

        except Exception as e:
            # Handle errors and retry
            retry += 1
            if retry > max_retry:
                return f"GPT3 error: {e}"
            print('Error communicating with OpenAI:', e)
            sleep(1)


# Define a function for GPT-3.5 Turbo completions
def gpt3_0613_completion(prompt, model='gpt-3.5-turbo-0613', messages=None, temperature=0.6, top_p=1.0, max_tokens=2000,
                         directory=None):

    # Attempt GPT-3.5 Turbo completion with retries
    max_retry = 5
    retry = 0
    while True:
        try:
            # Create a chat conversation with GPT-3.5 Turbo
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages or [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "assistant", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=['<<END>>']
            )
            try:
                # Extract and save the response
                text = response.choices[0].message.content.strip()
                filename = f'{time()}_gpt3.txt'
                with open(f'{directory}/{filename}', 'w') as outfile:
                    outfile.write('PROMPT:\n\n' + prompt + '\n\n====== \n\nRESPONSE:\n\n' + text)
                return text
            except:
                print("error saving to log")
        except Exception as oops:
            # Handle errors and retry
            retry += 1
            if retry > max_retry:
                return f"GPT3 error: {oops}"
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def extract_non_subject_pronouns(nlp, text):
    # Process the German text using the provided spaCy pipeline
    doc = nlp(text)

    # Find the subject of the sentence
    subject = None
    for token in doc:
        if "subj" in token.dep_:
            subject = token
            break

    # Extract pronouns (PRON) other than the subject and "ich"
    non_subject_pronouns = [token.text for token in doc if
                            token.pos_ == "PRON" and token != subject and token.text.lower() != "ich"]

    return non_subject_pronouns


# Define a function to open and read a file
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(file_content, destination_file):
    with open(destination_file, 'w', encoding='utf-8') as outfile:
        outfile.write(file_content)


import re
def remove_history(text, pattern_to_replace, pattern_to_add):
    pattern = r"CHAT-HISTORY:(.*?)<<history>>"
    return re.sub(pattern_to_replace, pattern_to_add, text, flags=re.DOTALL).strip()

import tiktoken
def num_tokens_from_string(text, encoding) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def split_chat_history(chat_history):
    pairs = chat_history.split('CHAT-HISTORY:\n')
    human_ai_pairs = []
    for pair in pairs:
        messages = pair.strip().split('\n')
        if len(messages) >= 2:
            human_ai_pairs.append(messages[-2] + '\n' + messages[-1])
    return human_ai_pairs

# Update the chat history with the new pair
def update_chat_history(chat_history, new_pair):
    pairs = split_chat_history(chat_history)
    if len(pairs) >= 4:
        pairs.pop(0)  # Remove the oldest pair
    pairs.append(new_pair)
    updated_chat_history = 'CHAT-HISTORY:\n' + '\nCHAT-HISTORY:\n'.join(pairs)
    return updated_chat_history

def replace_links_with_placeholder(text):
    # Define a regular expression pattern to match URLs
    url_pattern = r'https://\S+'
    modified_link = re.sub(url_pattern, '<<link>>', str(text))
    return modified_link

import smtplib
import ssl
from email.message import EmailMessage

def send_email(query):
    email_body = query
    email_sender = "kimberly.kent.twitter@gmail.com"
    email_password = "mihtepktfliycluz"
    email_receiver = "mefabe7562@marksia.com"
    subject = "Question from Websupportbot"

    em = EmailMessage()
    em["From"] = email_sender
    em["To"] = email_receiver
    em["Subject"] = subject
    em.set_content(email_body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
