import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
# import tfidf

def get_bert_embedding(sentence):
    model_name_or_path = "bert-base-multilingual-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    # Tokenize input text
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # Get BERT output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the output of the [CLS] token (pooled_output)
    bert_embedding = outputs.pooler_output.squeeze(0).cpu().numpy()
    return bert_embedding

def get_ditto_embedding(sentence, model_name_or_path, pooler='cls', layer=0, head=0):
    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tokenize input text
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)

    # Get raw embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        if hasattr(outputs, 'pooler_output'):
            pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        attention = outputs.attentions
        attention_diag = torch.diagonal(attention[layer][:, head, :, :], 0, dim1=1, dim2=2)

    # Apply different poolers
    if pooler == 'cls':
        # There is a linear+activation layer after CLS representation
        embedding = pooler_output.cpu().numpy()
    elif pooler == 'cls_before_pooler':
        embedding = last_hidden[:, 0].cpu().numpy()
    elif pooler == "avg":
        embedding = ((last_hidden * inputs['attention_mask'].unsqueeze(-1)).sum(1) / inputs['attention_mask'].sum(
            -1).unsqueeze(-1)).cpu().numpy()
    elif pooler == "avg_first_last":
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * inputs['attention_mask'].unsqueeze(-1)).sum(1) / inputs[
            'attention_mask'].sum(-1).unsqueeze(-1)
        embedding = pooled_result.cpu().numpy()
    elif pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 * inputs['attention_mask'].unsqueeze(-1)).sum(1) / \
                        inputs['attention_mask'].sum(-1).unsqueeze(-1)
        embedding = pooled_result.cpu().numpy()
    elif pooler == "att_first_last":
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * inputs['attention_mask'].unsqueeze(
            -1) * attention_diag.unsqueeze(-1)).sum(1)
        embedding = pooled_result.cpu().numpy()
    elif pooler == "att_last":
        pooled_result = (last_hidden * inputs['attention_mask'].unsqueeze(-1) * attention_diag.unsqueeze(-1)).sum(1)
        embedding = pooled_result.cpu().numpy()
    elif pooler == "att_static":
        with torch.no_grad():
            last_hidden = model.embeddings.word_embeddings(inputs['input_ids'])
        pooled_result = (last_hidden * inputs['attention_mask'].unsqueeze(-1) * attention_diag.unsqueeze(-1)).sum(1)
        embedding = pooled_result.cpu().numpy()
    elif pooler == "avg_static":
        with torch.no_grad():
            last_hidden = model.embeddings.word_embeddings(inputs['input_ids'])
        pooled_result = (last_hidden * inputs['attention_mask'].unsqueeze(-1)).sum(1) / inputs['attention_mask'].sum(
            -1).unsqueeze(-1)
        embedding = pooled_result.cpu().numpy()
    else:
        raise NotImplementedError
    return embedding

def compute_similarity(embedding1, embedding2):
    # Compute cosine similarity between two embeddings
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0, 0]
    return similarity

def main(sentence1, sentence2):
    # Get BERT embedding for each sentence
    bert_embedding1 = get_bert_embedding(sentence1)
    bert_embedding2 = get_bert_embedding(sentence2)

    # Get Ditto embedding for each sentence
    ditto_embedding1 = get_ditto_embedding(sentence1, model_name_or_path="bert-base-multilingual-uncased")
    ditto_embedding2 = get_ditto_embedding(sentence2, model_name_or_path="bert-base-multilingual-uncased")

    # Compute cosine similarity between BERT embeddings
    bert_similarity = compute_similarity(bert_embedding1, bert_embedding2)

    # Compute cosine similarity between Ditto embeddings
    ditto_similarity = compute_similarity(ditto_embedding1, ditto_embedding2)

    return bert_similarity, ditto_similarity

if __name__ == "__main__":
    sentence1 = "This is a test sentence."
    sentence2 = "This is another test sentence."
    bert_similarity, ditto_similarity = main(sentence1, sentence2)
    print("Cosine Similarity (BERT):", bert_similarity)
    print("Cosine Similarity (Ditto):", ditto_similarity)
