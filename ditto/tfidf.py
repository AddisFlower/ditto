import gensim
from gensim import corpora, models
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from gensim.corpora import MmCorpus
import numpy as np
import json

# This method is used to create the list of sentences that will be the corpus
def create_corpus(file_name):
    with open(file=file_name, mode='r', encoding='utf-8') as file:
        obj = file.read()
        sentences_list = obj.splitlines()
        return sentences_list

# Returns dictionary of token importance for each distinct token in the corpus. The dictionary is saved in a json file, so that token importance retrieval later on can be more efficient.
def token_importance(tfidf, dictionary, corpus, file_name):

    # Calculate the average TF-IDF score for each token in the dictionary
    token_tfidf_sum = {dictionary[id]: 0 for id in range(len(dictionary))}
    token_document_count = {dictionary[id]: 0 for id in range(len(dictionary))}

    for doc in corpus:
        for id, score in tfidf[doc]:
            token_tfidf_sum[dictionary[id]] += score
            token_document_count[dictionary[id]] += 1

    average_tfidf = {token: (token_tfidf_sum[token] / token_document_count[token]) 
                    for token in token_tfidf_sum if token_document_count[token] > 0}
    with open(file_name, 'w') as json_file:
        json.dump(average_tfidf, json_file)


language = 'pt'

# Create the sentence list using the 1 million wikepedia sentences
wiki_file_name = f'sampled_{language}_sentences.txt'
sentences_list = create_corpus(wiki_file_name)

# preprocess the sentences
custom_filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
processed_docs = [preprocess_string(doc, custom_filters) for doc in sentences_list]

# prepare a dictionary and a corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# create a TF-IDF model
tfidf = models.TfidfModel(corpus)  # fit model

file_name = f'average_tfidf_{language}.json'

# Creates and returns the average token importance json file. 
token_importance(tfidf, dictionary, corpus, file_name)
