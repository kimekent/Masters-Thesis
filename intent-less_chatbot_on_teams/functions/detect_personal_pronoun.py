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


import spacy

def find_coreferences(text):
    nlp = spacy.load("de_core_news_sm")
    doc = nlp(text)

    coreferences = {}

    for token in doc:
        # Check for pronouns that refer to a previous entity
        if token.dep_ == "prn" and token.head.i > 0:
            previous_token = token.head
            while previous_token.i > 0 and previous_token.dep_ != "root":
                previous_token = previous_token.head

            if previous_token.text not in coreferences:
                coreferences[previous_token.text] = []

            coreferences[previous_token.text].append(token.text)

    return coreferences

from transformers import pipeline

nlp = pipeline("coref-resolution", model="gsarti/coref-roberta-large")

text = "Was ist ein Anker? Wie kann man so einen einfügen."
coreferences = nlp(text)

for cluster in coreferences[0]["clusters"]:
    main_mention = cluster[0]
    mentions = cluster[1:]
    print(f"'{main_mention}' is referred to by: {', '.join(mentions)}")


