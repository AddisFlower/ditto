from datasets import load_dataset
import random
from nltk.tokenize import sent_tokenize

language = 'it'

# If the language is French
if language == 'fr':
    # Load the French Wikipedia dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.fr", split='train[:0.5%]')


    # Function to extract sentences from a text
    def extract_sentences(text):
        # This can be more complex and handle edge cases better with libraries like spaCy
        sentences = sent_tokenize(text, language='french')
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    # # Apply the function to extract sentences
    sentences = [sentence for text in dataset['text'] for sentence in extract_sentences(text)]

    # # Sample 1,000,000 sentences randomly
    sampled_sentences = random.sample(sentences, 1000000)
    # print(len(dataset))

    file_name = 'sampled_french_sentences.txt'
    # Save the sentences to a file
    with open(file_name, "w", encoding="utf-8") as file:
        for sentence in sampled_sentences   :
            file.write(sentence + "\n")

# If the language is Spanish
if language == 'es':
    # Load the spanish Wikipedia dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split='train[:0.5%]')

    # Function to extract sentences from a text
    def extract_sentences(text):
        # This can be more complex and handle edge cases better with libraries like spaCy
        sentences = sent_tokenize(text, language='spanish')
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    # # Apply the function to extract sentences
    sentences = [sentence for text in dataset['text'] for sentence in extract_sentences(text)]

    # # Sample 1,000,000 sentences randomly
    sampled_sentences = random.sample(sentences, 1000000)
    # print(len(dataset))

    file_name = 'sampled_spanish_sentences.txt'
    # Save the sentences to a file
    with open(file_name, "w", encoding="utf-8") as file:
        for sentence in sampled_sentences   :
            file.write(sentence + "\n")

# If the language is Italian
if language == 'it':
    # Load the spanish Wikipedia dataset
    dataset = load_dataset("wikipedia", "20220301.it", split='train[:5%]')

    # Function to extract sentences from a text
    def extract_sentences(text):
        # This can be more complex and handle edge cases better with libraries like spaCy
        sentences = sent_tokenize(text, language='italian')
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    # # Apply the function to extract sentences
    sentences = [sentence for text in dataset['text'] for sentence in extract_sentences(text)]

    # # Sample 1,000,000 sentences randomly
    sampled_sentences = random.sample(sentences, 1000000)
    # print(len(dataset))

    file_name = 'sampled_italian_sentences.txt'
    # Save the sentences to a file
    with open(file_name, "w", encoding="utf-8") as file:
        for sentence in sampled_sentences   :
            file.write(sentence + "\n")


