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
chroma_directory = path + r'\gpt3_gpt4_generation_comparison\testing-websupport-and-webhelp-db'
prompt_logs_directory = path + r"\logs\answer_generation"
retriever_prompt_log_directory = path + r"\logs\retriever"

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = open_file(path + r"\openaiapikey.txt")
openai.api_key = os.getenv("OPENAI_API_KEY")

vectordb_websupport_bot = Chroma(persist_directory=chroma_directory,
                                 embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002'))

words_to_check = ast.literal_eval(open_file(path + r"\testing_data\words_to_check_for_adjusted_similarity_score.txt"))

# 2. Chatbot------------------------------------------------------------------------------------------------------------
def main(question, generation_model):

    # Load prompts
    # This prompts is used to restructure question for the retriever. For each new session the history is deleted.
    retriever_prompt = remove_history(open_file(path + r"\testing_prompts\retriever\instruction_based_prompting.txt"),
                                      "<<newest message>>(.*?)<<oldest message>>",
                                      "<<newest message>>\n<<history>>\n<<oldest message>>")

    # This prompt is used to answer to question. For each new session the history is deleted.
    question_prompt = remove_history(open_file(path + r"\testing_prompts\answer_generation\4_final_prompt.txt"),
                                     "HISTORY:(.*?)<<history>>",
                                     "HISTORY: <<history>>")

    query_count = 0  # Load trackers, Initialize query count

    l_retriever_history = []  # This list will contain the previous questions and answers.
    l_qa_history = []  # This list will contain the previous questions and answers for the Q&A prompt.

    query = question  # Prompt the user for a question
    print("query: " + query)
    query_count += 1
    print("query count: " + str(query_count))

    if query_count > 1: #
        # Reformulate the current question to include context from previous turns for better document retrieval
        # in multi-question sessions.
        current_retriever_prompt = retriever_prompt.replace('<<query>>', query)
        restructured_query = gpt3_0613_completion(current_retriever_prompt,
                                                  directory=retriever_prompt_log_directory)
        print("restructured query: " + restructured_query)
    else:
        restructured_query = query

    # Initialize lists per question.
    l_webhelp_articles = []  # This list will contain all retrieved webhelp articles.
    l_webhelp_questions = []  # This list will contain all retrieved websupport questions.

    # Count how many tokens are in the prompt. In the beginning the token count only includes the initial prompt.
    total_tokens = num_tokens_from_string(question_prompt, encoding="cl100k_base")

    # Perform retrieval based on the user's query
    results = vectordb_websupport_bot.similarity_search_with_score(restructured_query, k=50)

    first_document_score = results[0][1]  # Extract the score of the first document

    # If cosine distance is below 0.3 continue answering question. Else hand off to support representative.
    if first_document_score < 0.3:
        results = adjust_similarity_scores_final_model_test(results, question=restructured_query, word_intent_dict=words_to_check,
                                           multiplier=0.8)
        print("results after adjustment: " + str(results[:10]))
        for doc in results:
            doc = doc[0]
            if doc.metadata.get("Source") == "webhelp-article":
                link = doc.metadata.get("Link")
                webhelp_article_content = doc.page_content
                context = f" {webhelp_article_content}\nLink: {link}"
                l_webhelp_articles.append(context)
                # print("l_webhelp_articles: " + str(l_webhelp_articles))

            elif doc.metadata.get("Source") == "websupport question":
                websupport_question = doc.page_content
                websupport_answer = doc.metadata.get("Answer", "No answer found")
                # Format the question and answer
                context = f"Q: {websupport_question}\nA: {websupport_answer}"
                l_webhelp_questions.append(context)

            # Get the number of tokens
            tokens = num_tokens_from_string(context, encoding="cl100k_base")

            # print("total tokens = " + str(total_tokens))

            # If adding the answer would exceed the token limit, break out of the loop.
            if total_tokens + tokens > 10000:
                break
            else:
                total_tokens += tokens

        # Construct a prompt for GPT-3.5 Turbo based on the user's question
        current_prompt = question_prompt \
            .replace('<<query>>', restructured_query) \
            .replace('<<websupport_questions>>', "\n".join(l_webhelp_questions)) \
            .replace("<<webhelp_article>>", " ".join(l_webhelp_articles))

        # print("prompt: " + current_prompt)

        # Generate answer to prompt
        response = gpt3_1106_completion(prompt=current_prompt, model=generation_model,  log_directory=path + r"\logs\answer_generation",
                                        max_tokens=1000)
        # print("response: " + response)

        # Add memory to retriever
        # Count how many tokens the retriever prompt has
        tokens_retriever = num_tokens_from_string(retriever_prompt, encoding="cl100k_base")
        if tokens_retriever > 3000:  # If token limit is reached, delete latest conversation turn
            l_retriever_history = l_retriever_history[1:]

        # Reverse list, so newest chat history is added to the top of prompt
        l_reversed_retriever_history = l_retriever_history[::-1]

        # Delete old history and add new history with the latest Q&A to prompt
        testing_retriever_prompt = remove_history(retriever_prompt,
                                                  "<<newest message>>(.*?)<<oldest message>>",
                                                  "<<newest message>>\n<<history>>\n<<oldest message>>")

        updated_retriever_prompt = testing_retriever_prompt.replace('<<history>>',
                                                                    "\n".join(l_reversed_retriever_history))

        # Save history to retriever prompt.
        save_file(updated_retriever_prompt, path + r"\testing_prompts\retriever\instruction_based_prompting.txt")

        # Add memory to Q&A prompt
        # In order not to exceed token length, only the last two conversation turns are added as history
        # to question_prompt.

        # Get last two conversation turns
        if len(l_qa_history) > 3:
            l_qa_history = l_qa_history[1:]

        last_QA_pair = f"\nHuman: {restructured_query} \nAI: {response}"
        l_qa_history.append(last_QA_pair)

        # print("prompt_history: " + " ".join(l_qa_history))
        # Delete all conversation turns
        question_prompt = remove_history(question_prompt,
                                         "Human:(.*?)<<history>>",
                                         "<<history>>")
        # Add new conversation turns
        question_prompt = question_prompt.replace('<<history>>', " ".join(l_qa_history) + "\n<<history>>")
        # print("updated_history: " + question_prompt)
    else:
        response = "Leider kann ich deine Frage nicht beantworten. " \
                   "Soll ich eine Websupport-Ticket mit deiner Frage eröffnen? (Antworte mit 'Ja' oder 'Nein')"

    print(response)
    return response

if __name__ == "__main__":
    main()