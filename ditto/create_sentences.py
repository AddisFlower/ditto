from datasets import load_dataset
import random
from nltk.tokenize import sent_tokenize
import numpy as np


language = 'pt'

# If the language is French
if language == 'fr':
    # Load the French Wikipedia dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.fr", split='train[:0.5%]')

    # Function to extract sentences from a text
    def extract_sentences(text):
        sentences = sent_tokenize(text, language='french')
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    # Extract sentences
    sentences = [sentence for text in dataset['text'] for sentence in extract_sentences(text)]

    # Sample 1,000,000 sentences randomly
    sampled_sentences = random.sample(sentences, 1000000)
    # print(len(dataset))

    file_name = 'sampled_fr_sentences.txt'
    # Save the sentences to a file
    with open(file_name, "w", encoding="utf-8") as file:
        for sentence in sampled_sentences   :
            file.write(sentence + "\n")

# If the language is Spanish
if language == 'es':
    # Load the spanish Wikipedia dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split='train', streaming=True)

    # Function to extract sentences from a text
    def extract_sentences(text):
        return sent_tokenize(text, language='spanish')

    # Collect 2,000,000 sentences to randomly sample 1,000,000 later
    sentences = []
    for sample in dataset.take(10000):  
        article_sentences = extract_sentences(sample['text'])
        sentences.extend(article_sentences)
        if len(sentences) > 2000000:  
            break
    # Ensure that you have at least 1,000,000 sentences
    if len(sentences) >= 1000000:
        random_indices = np.random.choice(len(sentences), 1000000, replace=False)
        sampled_sentences = [sentences[i] for i in random_indices]
    else:
        print("Not enough sentences collected, consider increasing the number of articles processed.")
    with open('sampled_es_sentences.txt', 'w', encoding='utf-8') as f:
        for sentence in sampled_sentences:
            f.write(sentence + '\n')

# If the language is Italian
if language == 'it':
    # Load the Italian Wikipedia dataset
    dataset = load_dataset("wikipedia", "20220301.it", split='train[:5%]')

    # Function to extract sentences from a text
    def extract_sentences(text):
        sentences = sent_tokenize(text, language='italian')
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    # Extract sentences
    sentences = [sentence for text in dataset['text'] for sentence in extract_sentences(text)]

    # Sample 1,000,000 sentences randomly
    sampled_sentences = random.sample(sentences, 1000000)

    file_name = 'sampled_it_sentences.txt'
    # Save the sentences to a file
    with open(file_name, "w", encoding="utf-8") as file:
        for sentence in sampled_sentences   :
            file.write(sentence + "\n")

# If the language is Portuguese
if language == 'pt':
    # Load the portuguese Wikipedia dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.pt", split='train', streaming=True)

    # Function to extract sentences from a text
    def extract_sentences(text):
        return sent_tokenize(text, language='portuguese')

    sentences = []
    # Collect 2,000,000 sentences to randomly sample 1,000,000 later
    for sample in dataset.take(30000):
        print(i)
        article_sentences = extract_sentences(sample['text'])
        sentences.extend(article_sentences)
        if len(sentences) > 2000000:  # Collect more to ensure randomness in final sample
            break
    # Ensure that you have at least 1,000,000 sentences
    if len(sentences) >= 1000000:
        random_indices = np.random.choice(len(sentences), 1000000, replace=False)
        sampled_sentences = [sentences[i] for i in random_indices]
    else:
        print("Not enough sentences collected, consider increasing the number of articles processed.")
    with open('sampled_pt_sentences.txt', 'w', encoding='utf-8') as f:
        for sentence in sampled_sentences:
            f.write(sentence + '\n')
