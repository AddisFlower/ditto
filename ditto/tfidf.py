import gensim
from gensim import corpora, models
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from gensim.corpora import MmCorpus
import numpy as np
import json

def get_token_tfidf(token, dictionary, tfidf_model, corpus):
    token_id = dictionary.token2id.get(token)
    if token_id is None:
        return None
    
    # Iterate over the corpus to get TF-IDF scores for the token
    token_tfidf_scores = []
    for doc_id, doc in enumerate(corpus):
        for id, score in tfidf_model[doc]:
            if id == token_id:
                token_tfidf_scores.append((doc_id, score))
                break
    # Return the average importance of a token accross all documents                
    return sum(score for _, score in token_tfidf_scores) / len(token_tfidf_scores)

# This method is used to create the list of sentences that will be the corpus
def create_corpus(file_name):
    with open(file=file_name, mode='r', encoding='utf-8') as file:
        obj = file.read()
        sentences_list = obj.splitlines()
        return sentences_list
    
def save_model(tfidf, dictionary, corpus):
    tfidf.save('tfidf_model.tfidf')
    dictionary.save('dictionary.dict')
    MmCorpus.serialize('my_corpus.mm', corpus)

def load_and_use_model():
    loaded_tfidf = models.TfidfModel.load('tfidf_model.tfidf')
    loaded_dictionary = corpora.Dictionary.load('dictionary.dict')
    loaded_corpus = MmCorpus('my_corpus.mm')
    return loaded_tfidf, loaded_dictionary, loaded_corpus

# Returns dictionary of token importance for each distinct token in the corpus. The dictionary is saved in a json file, so that token importance retrieval later on can be more efficient.
def token_importance(tfidf, dictionary, corpus):

    # Calculate the average TF-IDF score for each token in the dictionary
    token_tfidf_sum = {dictionary[id]: 0 for id in range(len(dictionary))}
    token_document_count = {dictionary[id]: 0 for id in range(len(dictionary))}

    for doc in corpus:
        for id, score in tfidf[doc]:
            token_tfidf_sum[dictionary[id]] += score
            token_document_count[dictionary[id]] += 1

    average_tfidf = {token: (token_tfidf_sum[token] / token_document_count[token]) 
                    for token in token_tfidf_sum if token_document_count[token] > 0}
    with open('average_tfidf.json', 'w') as json_file:
        json.dump(average_tfidf, json_file)

# The below code that has been commented was initially utilized to create the model. 

# # Create the sentence list using the 1 million wikepedia sentences
# wiki_file_name = 'wiki1m_for_simcse.txt'
# sentences_list = create_corpus(wiki_file_name)

# # preprocess the sentences
# custom_filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
# processed_docs = [preprocess_string(doc, custom_filters) for doc in sentences_list]

# # prepare a dictionary and a corpus
# dictionary = corpora.Dictionary(processed_docs)
# corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# # create a TF-IDF model
# tfidf = models.TfidfModel(corpus)  # fit model

# # apply transformation to the whole corpus
# # corpus_tfidf = tfidf[corpus]

# # Example of getting the average token importance for the token "and"
# print(get_token_tfidf("way", dictionary, tfidf, corpus))

# # # Save model for easier reprocessing
# save_model(tfidf, dictionary, corpus)


# Load the saved model
tfidf, dictionary, corpus = load_and_use_model()
# Creates and returns the average token importance json file. 
token_importance(tfidf, dictionary, corpus)

