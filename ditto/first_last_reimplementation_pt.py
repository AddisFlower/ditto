# Resources Utilized:
# https://github.com/alibaba-damo-academy/SpokenNLP/tree/main/ditto
# https://github.com/alibaba-damo-academy/SpokenNLP/tree/main/ditto/SentEval
# https://github.com/princeton-nlp/SimCSE?tab=readme-ov-file#overview
# https://realpython.com/python-string-concatenation/#efficiently-concatenating-many-strings-with-join-in-python

import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import sys
import logging
from prettytable import PrettyTable


# constants used to represent necessary paths for SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
BERT_IMPORTANCE_ATTENTION_LAYER = 9
BERT_IMPORTANCE_ATTENTION_HEAD = 12

# load SentEval toolkit
# note: the SimCSE paper authors made the following modifications to the SentEval toolkit
#       they added the "all" setting to all the STS tasks and
#       changed STS-B and SICK-R to not use an additional regressor
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# function used by the authors for output formatting
def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

# SentEval prepare (can be used to construct the vocabulary)
def prepare(params, samples):
    return

# note: batcher (transforms a batch of text sentences into sentence embeddings)
def batcher(params, batch):
    # below code snippet utilized by SimCSE authors to deal with rare token encoding issues within
    # the STS datasets
    if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
        # for testing if there are any encoding issues in the STS datasets
        print('Warning: Rare Encoding Issue Occurred')
        # handles token encoding issues by decoding the bytes object token a utf-8 string object
        batch = [[word.decode('utf-8') for word in s] for s in batch]

    # print(f'Number of Sentences in Batch: {len(batch)}')

    # concatenate the words in the sentences together again. 
    # this is required due to the load file method of the STSEval class in SentEval 
    # splitting the original examples from the dataset (consisting of two sentences per example)
    # into arrays of words to sort the examples by number of words in each sentence plus their semantic
    # similarity label, which helps with minimizing the necessary padding in this function
    sentences = [' '.join(sentence) for sentence in batch]

    # use the pretrained model's tokenizer to get encodings for each word in each sentence of the batch
    batch_encoded_dict = tokenizer.batch_encode_plus(sentences, 
                                                     add_special_tokens = True,
                                                     padding = True, 
                                                     return_tensors = 'pt')
    
    # batch_encoded_dict = batch_encoded_dict.to(device)
    input_ids = batch_encoded_dict['input_ids']
    token_type_ids = batch_encoded_dict['token_type_ids']
    attention_masks = batch_encoded_dict['attention_mask']

    b_input_ids = input_ids.to(device)
    b_token_type_ids = token_type_ids.to(device)
    b_attention_masks = attention_masks.to(device)

    # used to get hidden layers (which includes the final word embeddings) and attentions 
    # from the pretrained model by doing a forward pass using the batch's encoded sentences 
    # for later use in computing the sentence embeddings
    with torch.no_grad():
        # put the model in evaluation mode
        model.eval()
        outputs = model(input_ids=b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_attention_masks,
                        output_hidden_states=True,
                        output_attentions=True,
                        return_dict=True
                        )
        # tuple of tensors (one for the output of the embeddings + one for output of each layer) of shape
        # (batch_size, sequence_length, hidden_size)
        hidden_states = outputs['hidden_states']
        # tuple of tensors (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
        attentions = outputs['attentions']

    # # check dimensions of returned hidden_states and attentions tuples
    # print(f'Number of Layers: {len(hidden_states)}')
    # print(f'Number of Attention Layers: {len(attentions)}')

    # get the attention layer identified as representing the word importance in the paper
    importance_attention_layer = attentions[BERT_IMPORTANCE_ATTENTION_LAYER - 1]

    # # check dimensions of tensors in hidden_states and attentions
    # print(f'Dimensions of Tensors in hidden_states: {hidden_states[-1].size()}')
    # print(f'Dimensions of Tensors in Importance Attention Layer: {importance_attention_layer.size()}')

    # get the attention head matrix identified as representing the word importance in the paper
    importance_attention_heads = importance_attention_layer[:, BERT_IMPORTANCE_ATTENTION_HEAD - 1, :, :]

    # get the diagonal values of the attention matrix for each sentence in the batch
    diagonal_values = torch.diagonal(importance_attention_heads, 0, 1, 2)

    # get the last_hidden_layer and first_hidden_layer from hidden_states for computing BERT first-last Ditto
    last_hidden_layer = hidden_states[-1]
    first_hidden_layer = hidden_states[0]

    # # check dimensions of tensors in importance_attention_heads and diagonal_values
    # # print(f'Dimensions of Tensors in Importance Attention Heads: {importance_attention_heads.size()}')
    # print(f'Dimensions of Tensors in Diagonal Attention Values: {diagonal_values.size()}')
    # print(f'Dimensions of attention_masks {attention_masks.size()}')
    # print(f'Dimensions of last_hidden_layer {last_hidden_layer.size()}')
    # print(f'Dimensions of first_hidden_layer {first_hidden_layer.size()}')

    # the below lines of code are heavily inspired by the authors, especially the utilization of the unsqueeze()
    # function to allow for matrix multiplication 
    # compute sentence embeddings for BERT first-last Ditto
    # not normalized by N due to potentially resulting in very small values
    # (1/2) * Summation_{i=1 to N} (A_{ii} * ((h_{i L} + h_{i 1}) * attention_masks))
    # note: the first-last hidden layer is multiplied by attention_masks to zero out the 
    # embeddings for padding tokens

    # compute sum of first_hidden_layer and last_hidden_layer
    first_last_hidden_layer = last_hidden_layer + first_hidden_layer

    # # for debugging
    # print(last_hidden_layer[0])
    # print(first_hidden_layer[0])
    # print(first_last_hidden_layer[0])

    # append a singleton dimension to attention_masks as the last dimension
    # this is required for matrix multiplication due to PyTorch's broadcasting semantics, which
    # allows for batched matrix multiplication
    broadcastable_attention_masks = b_attention_masks.unsqueeze(-1)

    # compute the first_last_hidden_layer with its embeddings for padding words zero'd out
    first_last_hidden_padding_zero = first_last_hidden_layer * broadcastable_attention_masks

    # # for debugging
    # print(b_attention_masks[0])
    # print(first_last_hidden_layer[0])
    # print(first_last_hidden_padding_zero[0])

    # append a singleton dimension to diagonal_values for the same reasoning as above
    broadcastable_diagonal_values = diagonal_values.unsqueeze(-1)

    ditto_word_embeddings = first_last_hidden_padding_zero * broadcastable_diagonal_values

    # # for debugging
    # print(broadcastable_diagonal_values[0])
    # print(first_last_hidden_padding_zero[0])
    # print(ditto_word_embeddings[0])

    # compute the sentence embeddings by summing over each word in the sentence's 
    # ditto word embeddings
    sentence_embeddings = torch.sum(ditto_word_embeddings, dim=1)

    # # for debugging
    # print(ditto_word_embeddings.size())
    # print(sentence_embeddings.size())
    # print(ditto_word_embeddings[0].sum(0))
    # print(sentence_embeddings[0])

    return sentence_embeddings.cpu()

torch.cuda.empty_cache()

# Confirm that the GPU is detected
assert torch.cuda.is_available()

# Get the GPU device name.
device_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"Found device: {device_name}, n_gpu: {n_gpu}")
device = torch.device("cuda")

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# the parameters used by the authors for senteval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                        'tenacity': 5, 'epoch_size': 4}

# # load model checkpoints
# model = AutoModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# load model checkpoints for multilingual model
model = AutoModel.from_pretrained("bert-base-multilingual-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

# tell pytorch to run the model on GPU
model.cuda()

# for multilingual testing on translated STSBenchmark datasets (ignore avg spearman correlation)
sts_tasks = ['STSBenchmarkPT']

# creates an object of the class associated with the task, which loads the required datasets,
# then initializes the created object's similarity field and runs the prepare function.
# finally, runs the batcher function to get the sentence-level encodings for each batch and computes the
# pearson and spearman scores for all the datasets associated with the task
se = senteval.engine.SE(params_senteval, batcher, prepare)
results = se.eval(sts_tasks)

# code provided by the authors to retrieve the spearman correlation results for all datasets of the 
# 'STS' prefixed tasks except for STSBenchmark. for STSBenchmark and SICKRelatedness, the code retrieves
# the spearman correlation results for the test datasets.
# also prepares the output table to showcase the experiment results
print("------ %s ------" % ('test'))
task_names = []
scores = []
for task in sts_tasks:
    task_names.append(task)
    if task in results:
        if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
        else:
            scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
    else:
        scores.append("0.00")
task_names.append("Avg.")
scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
print_table(task_names, scores)
print(' '.join(scores))