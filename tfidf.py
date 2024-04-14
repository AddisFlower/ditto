import gensim
from gensim import corpora, models
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
import numpy as np

# This method is used to create the list of sentences that will be the corpus
def create_corpus(file_name):
    with open(file=file_name, mode='r', encoding='utf-8') as file:
        obj = file.read()
        sentences_list = obj.splitlines()
        return sentences_list

wiki_file_name = 'wiki1m_for_simcse.txt'
sentences_list = create_corpus(wiki_file_name)

# Step 1: Preprocess the documents
# Custom preprocessing pipeline
custom_filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
processed_docs = [preprocess_string(doc, custom_filters) for doc in sentences_list]

# Step 2: Prepare a dictionary and a corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Step 3: Create a TF-IDF model
tfidf = models.TfidfModel(corpus)  # fit model

# Step 4: Apply transformation to the whole corpus
corpus_tfidf = tfidf[corpus]

# # Output the TF-IDF weights
# for doc in corpus_tfidf:
#     print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])


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

print(get_token_tfidf("and", dictionary, tfidf, corpus))