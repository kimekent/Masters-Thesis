"""
This Python script is designed for testing and evaluating a chatbot's performance in information retrieval.

It includes several key components and steps:

- Building Retrievers: The script features two types of retrievers: One using BERT embeddings and
    the other using OpenAI embeddings (step 2).

- Testing Retrievers: The script tests the retrievers on a set of websupport queries (step 3).

- Evaluation of Retrieval Effectiveness: It evaluates the effectiveness of the retrievers by assessing how often
    they retrieve correct documents (step 5).

- Visualization: The results are visualized using bar charts to represent the question answerability rate by
    document count for each retriever (step 6).

To visualize the results the script can be run from step 5 - 6 using the saved result csvs.
"""
# 1. Set Up-------------------------------------------------------------------------------------------------------------
# Standard Libraries
import pandas as pd
import json
import os

# Libraries for running the testing chatbot / for text generation
import importlib
from sentence_transformers import SentenceTransformer
import openai

# Own functions
from testing_functions import replace_links_with_placeholder, open_file, num_tokens_from_string, adjust_similarity_scores

# Libraries for transforming csv data
import ast

# Libraries for visualization
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Set Variables and paths
path = r"C:\Users\Kimberly Kent\Documents\Master\HS23\Masterarbeit\testing"
openai.api_key = open_file(path + r'\openaiapikey.txt')


# 2. Build the retrievers-----------------------------------------------------------------------------------------------
"""The retriever encodes the input question into a numerical vector using a predefined encoder.
It then calculates the similarity score between the encoded question vector and each document
vector in the loaded index using the dot product.
The retriever returns the top 'n' documents with the highest similarity scores"""

words_to_check = ast.literal_eval(open_file(path + r"\testing_data\words_to_check_for_adjusted_similarity_score.txt"))

# Function calculating the dot product.
def similarity(v1, v2):  # returns dot product of two vectors
    return np.dot(v1, v2)


# Defining the retriever that uses BERT embeddings, specifically the embeddings from the model
# 'sentence-transformers/facebook-dpr-question_encoder-single-nq-base' which is a DPR model
def BERT_retriever(query=None, count=20, index_path=None):
    question_model = SentenceTransformer('sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
    vector = question_model.encode(query)
    with open(index_path, 'r') as infile:
        data = json.load(infile)
    scores = []
    for i in data:
        score = similarity(vector, i['vector'])
        scores.append({'context': i['context'], 'score': score, "metadata": i["metadata"]})
    # Comment for unadjusted scores: If certain words or word combinations in the question indicate a specific intent,
    # the scores of documents associated with that intent are multiplied.
    # This increases the likelihood of selecting documents relevant to the question.
    scores = adjust_similarity_scores(results=scores, question=query, word_intent_dict=words_to_check, multiplier=1.2)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    return ordered[0:count]



# This function is used to get the embeddings form the OpenAI model 'text-embedding-ada-002'
def gpt3_embedding(content, engine="text-embedding-ada-002"):
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


# Defining the retriever using OpenAI embeddings, specifically the embeddings from the OpenAI model
# 'text-embedding-ada-002'
def OpenAI_retriever(query=None, count=20, index_path=None):
    vector = gpt3_embedding(query)
    with open(index_path, 'r') as infile:
        data = json.load(infile)
    scores = []
    for i in data:
        score = similarity(vector, i['vector'])
        scores.append({'context': i['context'], 'score': score, "metadata": i["metadata"]})
    # Comment for unadjusted scores: If certain words or word combinations in the question indicate a specific intent,
    # the scores of documents associated with that intent are multiplied. This increases
    # the likelihood of selecting documents relevant to the question.
    scores = adjust_similarity_scores(results=scores, question=query, word_intent_dict=words_to_check, multiplier=1.2)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    return ordered[0:count]


# 3. Use retrievers to retrieve a set of 'n' documents for each question in the testing dataset-------------------------
# Import testing dataset to test the two retrievers on
testing_data_set = pd.read_csv(path + r"\testing_data\test_dataset.csv", encoding="utf-8")

questions = testing_data_set["Beschreibung"].tolist()
# In the index links were replaced with placeholders. To maintain consistency, the same is done to these questions.
questions = [replace_links_with_placeholder(i) for i in questions]
human_responses = testing_data_set["Lösungsbeschreibung"].tolist()
question_ids = testing_data_set["Incident-ID"].tolist()
intents = testing_data_set["intent"].tolist()

# Import OpenAI testing chatbot to test the retriever component.
"""This testing chatbot is specifically designed for evaluating individual chatbot components.
It takes various parameters, including the test type, retriever selection, the chosen prompt,
and input size, to systematically assess the chatbot's performance."""

module = importlib.import_module("testing_OpenAI-Chatbot")

"""The retrievers were run multiple times. In each iteration the number of retrieved documents was increased
to find the optimal number of documents that needed to be returned in order to be able to answer the question."""
# Retrieve documents using the BERT retriever
# This loop was run seven times, varying the number of retrieved documents per question from 20 to 70,
# increasing by 10 each time.
l_results_BERT_retriever = []
l_retrieved_documents_content = []
l_references = []

for i, question in enumerate(questions):
    retrieved_documents = module.main(test="retriever", query=question,
                                      retriever=BERT_retriever(query=question, count=70,
                                                               index_path=path + r"\indexes\BERT_index.json"))
    for doc in retrieved_documents:

        if doc['metadata']['source'] == "webhelp_article":
            retrieved_document_content = doc['context']
            reference = doc["metadata"]["Link"]

        elif doc['metadata']['source'] == "websupport_question":
            retrieved_document_content = doc["metadata"]["answer"]
            reference = doc["metadata"]["question_id"]

        l_retrieved_documents_content.append(retrieved_document_content)
        l_references.append(reference)

    dic_BERT_retriever = {
        "question": question,
        "question_ID": question_ids[i],
        "intent": intents[i],
        'retrieved_document_content': l_retrieved_documents_content,
        'reference': l_references,
        "human_response": human_responses[i],
        "source": "BERT_retriever"
    }
    l_results_BERT_retriever.append(dic_BERT_retriever)
    l_retrieved_documents_content = []
    l_references = []

# Retrieve documents using the OpenAI retriever
# This loop was run five times, varying the number of retrieved documents per question from 10 to 50,
# increasing by 10 each time.
l_results_OpenAI_retriever = []
l_retrieved_documents_content = []
l_references = []
for i, question in enumerate(questions):
    retrieved_documents = module.main(test="retriever", query=question,
                                      retriever=OpenAI_retriever(query=question, count=70,
                                                                 index_path=path + r"\indexes\OpenAI_index.json"))
    for doc in retrieved_documents:
        if doc['metadata']['source'] == "webhelp_article":
            retrieved_document_content = doc['context']
            reference = doc["metadata"]["Link"]

        elif doc['metadata']['source'] == "websupport_question":
            retrieved_document_content = doc["metadata"]["answer"]
            reference = doc["metadata"]["question_id"]

        l_retrieved_documents_content.append(retrieved_document_content)
        l_references.append(reference)

    dic_OpenAI_retriever = {
        "question": question,
        "question_ID": question_ids[i],
        "intent": intents[i],
        'retrieved_document_content': l_retrieved_documents_content,
        'reference': l_references,
        "human_response": human_responses[i],
        "source": "OpenAI_retriever"
    }
    l_results_OpenAI_retriever.append(dic_OpenAI_retriever)
    l_retrieved_documents_content = []
    l_references = []

# 4. Save retrieved documents to csv file-------------------------------------------------------------------------------
"""Retrieved documents were saved to csv file for subsequent evaluation of metrics
such as corpus BLEU score, ROUGE-L score, and semantic similarity to assess their relevance against human responses."""

combined_list = l_results_OpenAI_retriever + l_results_BERT_retriever

df = pd.DataFrame(combined_list,
                  columns=['question', "question_ID", "intent", "retrieved_document_content", "reference",
                           "human_response", "source"])

df.to_csv(path + r"\testing_results\retriever_results\retriever_adjusted_scores_70.csv")

# 5. Find out how many times the correct document was retrieved---------------------------------------------------------
"""The core idea is to assess whether the retrieved documents are relevant to answer the question.
The result of this analysis is the number of answerable questions per retriever and number of retrieved documents.
A retrieval is considered successful, if at least one websupport answer, that has the same intent as
the current question is retrieved or if one webhelp article, that is in the list of relevant webhelp articles for that
question is retrieved. If either of these conditions are fulfilled, it is assumed that the question can be answered with 
the retrieved documents"""

# This data set contains websupport questions and answers. For each Q&A the intent and relevant webhelp articles are
# listed in separate columns
df_full_websupport_dataset = pd.read_csv(path + r"\testing_data\cleaned_websupport_questions_with_intents_utf-8.csv")

# These csv files contain a sample of the websupport questions and a list of either 20, 30, 40, 50, 60 or 70
# retrieved documents (websupport questions and webhelp articles) per question.
# To see results without adjusted retriever scores, uncomment retriever results csv
csv_files_adjusted_scores = [
    (path + r'\testing_results\retriever_results\retriever_adjusted_scores_20.csv', 20),
    (path + r'\testing_results\retriever_results\retriever_adjusted_scores_30.csv', 30),
    (path + r'\testing_results\retriever_results\retriever_adjusted_scores_40.csv', 40),
    (path + r'\testing_results\retriever_results\retriever_adjusted_scores_50.csv', 50),
    (path + r'\testing_results\retriever_results\retriever_adjusted_scores_60.csv', 60),
    (path + r'\testing_results\retriever_results\retriever_adjusted_scores_70.csv', 70)
]

csv_files_unadjusted = [
    (path + r'\testing_results\retriever_results\retriever_results_20.csv', 20),
    (path + r'\testing_results\retriever_results\retriever_results_30.csv', 30),
    (path + r'\testing_results\retriever_results\retriever_results_40.csv', 40),
    (path + r'\testing_results\retriever_results\retriever_results_50.csv', 50),
    (path + r'\testing_results\retriever_results\retriever_results_60.csv', 60),
    (path + r'\testing_results\retriever_results\retriever_results_70.csv', 70)
]

# Create dictionary to understand which questions (identified by their IDs) are associated with which intents.
lists_by_intent = {}
for intent, group in df_full_websupport_dataset.groupby('intent'):
    lists_by_intent[intent] = group['Incident-ID'].tolist()

# Create dictionary to understand which questions (identified by their IDs) can be answered with which webhelp article.
lists_by_ID = {}
for ID, group in df_full_websupport_dataset.groupby('Incident-ID'):
    combined_webhelp_articles = group['webhelp article'].tolist() + group['webhelp article 2'].tolist()
    lists_by_ID[ID] = combined_webhelp_articles

# Initialize a dictionary to store the results
answerable_questions_adjusted_scores = {retriever: {num_docs: 0 for _, num_docs in csv_files_adjusted_scores} for retriever in ['OpenAI', 'DPR']}
unadjusted_answerable_questions = {retriever: {num_docs: 0 for _, num_docs in csv_files_unadjusted} for retriever in ['OpenAI', 'DPR']}

"""Iterate through each csv containing the retrieved documents per question and count how many of the questions can
be answered with the retrieved docs. THis is done once for the adjusted similarity scores and once for the unadjusted
scores"""
for csv_file, num_docs in csv_files_adjusted_scores:
    df = pd.read_csv(csv_file)

    for i, row in df.iterrows():
        count = 0
        question_intent = row['intent']  # Get intent of question
        # get list of retrieved web support question IDs and webhelp article links.
        retrieved_documents = ast.literal_eval(row['reference'])

        for doc in retrieved_documents:
            # Check if the retrieved doc has an ID that is associated with same intent as the question
            # for which it retrieved the doc.
            if doc in lists_by_intent.get(question_intent, []):
                count += 1
            # Check if a webhelp article that is associated with that question was retrieved.
            if doc in lists_by_ID.get(df_full_websupport_dataset["Incident-ID"][i], []):
                count += 1
        # If at least one web support answer with the same intent as question or one webhelp article that is assosiated
        # to the question is retrieved, the amount of answerable questions goes up by one.
        if count >= 1 and row['source'] == "BERT_retriever":
            answerable_questions_adjusted_scores['DPR'][num_docs] += 1
        elif count >= 1 and row['source'] == "OpenAI_retriever":
            answerable_questions_adjusted_scores['OpenAI'][num_docs] += 1

# Convert the counts of relevant retrievals to percentages.
# The percentage represents the percentage of answerable questions for the retrieved docs
for retriever in answerable_questions_adjusted_scores:
    for num_docs in answerable_questions_adjusted_scores[retriever]:
        answerable_questions_adjusted_scores[retriever][num_docs] = (answerable_questions_adjusted_scores[retriever][num_docs] / total_questions) * 100

# Unadjusted Scores
for csv_file, num_docs in csv_files_unadjusted:
    df = pd.read_csv(csv_file)

    for i, row in df.iterrows():
        count = 0
        question_intent = row['intent']  # Get intent of question
        # get list of retrieved web support question IDs and webhelp article links.
        retrieved_documents = ast.literal_eval(row['reference'])

        for doc in retrieved_documents:
            # Check if the retrieved doc has an ID that is associated with same intent as the question
            # for which it retrieved the doc.
            if doc in lists_by_intent.get(question_intent, []):
                count += 1
            # Check if a webhelp article that is associated with that question was retrieved.
            if doc in lists_by_ID.get(df_full_websupport_dataset["Incident-ID"][i], []):
                count += 1
        # If at least one web support answer with the same intent as question or one webhelp article that is assosiated
        # to the question is retrieved, the amount of answerable questions goes up by one.
        if count >= 1 and row['source'] == "BERT_retriever":
            unadjusted_answerable_questions['DPR'][num_docs] += 1
        elif count >= 1 and row['source'] == "OpenAI_retriever":
            unadjusted_answerable_questions['OpenAI'][num_docs] += 1

# Convert the counts of relevant retrievals to percentages.
# The percentage represents the percentage of answerable questions for the retrieved docs
for retriever in unadjusted_answerable_questions:
    for num_docs in unadjusted_answerable_questions[retriever]:
        unadjusted_answerable_questions[retriever][num_docs] = (unadjusted_answerable_questions[retriever][num_docs] / total_questions) * 100

# 6. Visualizing the results--------------------------------------------------------------------------------------------
# Preparing data for plotting
labels = sorted(answerable_questions_adjusted_scores['OpenAI'].keys())
adjusted_DPR_counts = [answerable_questions_adjusted_scores['DPR'][label] for label in labels]
adjusted_openai_counts = [answerable_questions_adjusted_scores['OpenAI'][label] for label in labels]
unadjusted_DPR_counts = [unadjusted_answerable_questions['DPR'][label] for label in labels]
unadjusted_openai_counts = [unadjusted_answerable_questions['OpenAI'][label] for label in labels]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots()

# Plot the adjusted results
rects1 = ax.bar(x - width / 2, adjusted_DPR_counts, width, label='Adjusted DPR Retriever', color="#1a2639")
rects2 = ax.bar(x + width / 2, adjusted_openai_counts, width, label='Adjusted OpenAI Retriever', color="#c24d2c")

# Plot the unadjusted results
rects3 = ax.bar(x - width / 2, unadjusted_DPR_counts, width, label='Unadjusted DPR Retriever', alpha=0.5, color="#1f77b4", hatch='//')
rects4 = ax.bar(x + width / 2, unadjusted_openai_counts, width, label='Unadjusted OpenAI Retriever', alpha=0.5, color="#ff7f0e", hatch='//')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.tick_params(axis='y', labelsize=14)
ax.set_xlabel('Number of Retrieved Documents per Question', fontsize=16)
ax.set_ylabel('Question Answerability Rate', fontsize=16)
#ax.set_title('Question Answerability Rate by Document Count and Retriever', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)

ax.legend(fontsize=14)

# Adjust the figure size to make room for the legend
fig.set_size_inches(12, 6)
# Place the legend to the right of the plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
# Adjust the layout to take the new legend location into account
plt.tight_layout()

plt.show()
#-----------------------------------------------------------------------------------------------------------------------


# Print the answerability rates for each retriever and number of retrieved documents
print("Answerability Rates: " + str(unadjusted_answerable_questions))
for retriever, docs_counts in unadjusted_answerable_questions.items():
    print(docs_counts)
    print(f"\nRetriever: {retriever}")
    #for num_docs, rate in docs_counts.items():
    for num_docs, rate in docs_counts.items():
        print(f"  Documents Retrieved: {num_docs}, Answerability Rate: {rate:.2f}%")

total_num_tokens = 0
for websupport_doc in df_full_websupport_dataset["Lösungsbeschreibung"]:
    num_tokens = num_tokens_from_string(websupport_doc, encoding="cl100k_base")
    total_num_tokens += num_tokens
print(total_num_tokens / len(df_full_websupport_dataset))

count_files = 0
for filename in os.listdir(path + r"\testing_data\webhelp_articles"):
    if filename.endswith(".txt"):
        webhelp_article = open_file(path + r"\\testing_data\\webhelp_articles\\" + filename)
        num_tokens = num_tokens_from_string(webhelp_article, encoding="cl100k_base")
        total_num_tokens += num_tokens
        count_files += 1
print(total_num_tokens / count_files)

# Print the final results
for retriever, counts in answerable_questions_adjusted_scores.items():
    print(f"Results for {retriever}:")
    for num_docs, count in counts.items():
        print(f"  Number of documents retrieved: {num_docs}, Relevant retrievals: {count}")
    print()

total_questions = len(df) / 2