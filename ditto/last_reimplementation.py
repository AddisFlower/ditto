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


# constants used to represent necessary paths for SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
BERT_IMPORTANCE_ATTENTION_LAYER = 1
BERT_IMPORTANCE_ATTENTION_HEAD = 10

# load SentEval toolkit
# note: the SimCSE paper authors made the following modifications to the SentEval toolkit
#       they added the "all" setting to all the STS tasks and
#       changed STS-B and SICK-R to not use an additional regressor
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

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

    print(f'Number of Sentences in Batch: {len(batch)}')

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

    # move tensors to the GPU
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

    # check dimensions of returned hidden_states and attentions tuples
    print(f'Number of Layers: {len(hidden_states)}')
    print(f'Number of Attention Layers: {len(attentions)}')

    # get the attention layer identified as representing the word importance in the paper
    importance_attention_layer = attentions[BERT_IMPORTANCE_ATTENTION_LAYER - 1]

    # check dimensions of tensors in hidden_states and attentions
    print(f'Dimensions of Tensors in hidden_states: {hidden_states[-1].size()}')
    print(f'Dimensions of Tensors in Importance Attention Layer: {importance_attention_layer.size()}')

    # get the attention head matrix identified as representing the word importance in the paper
    importance_attention_heads = importance_attention_layer[:, BERT_IMPORTANCE_ATTENTION_HEAD - 1, :, :]

    # get the diagonal values of the attention matrix for each sentence in the batch
    diagonal_values = torch.diagonal(importance_attention_heads, 0, 1, 2)
    #

    # check dimensions of tensors in importance_attention_heads and diagonal_values
    print(f'Dimensions of Tensors in Importance Attention Heads: {importance_attention_heads.size()}')
    print(f'Dimensions of Tensors in Diagonal Attention Values: {diagonal_values.size()}')

    # compute sentence embedding for BERT last Ditto
    # not normalized by N due to potentially resulting in very small values
    # (1/2) * Summation_{i=1 to N} (A_{ii} * (h_{i L}))




    sentence_embeddings = None
    exit(1)
    
    return sentence_embeddings



    # # for testing
    # print(importance_attention_layer[0])
    # print(importance_attention_heads[0])
    # print(diagonal_values[0])
    # print(diagonal_values.size())

    # check dimension of tensors in importance_attention_heads
    # print(importance_attention_heads.size())

    # # print out the attention matrix for layer 1 and attention head 10
    # importance_attention_head = outputs['attentions'][BERT_IMPORTANCE_ATTENTION_LAYER - 1][BERT_IMPORTANCE_ATTENTION_HEAD - 1][0] # currently only looking at the first sentence in the batch

    # print(importance_attention_head)
    # print('testing here')
    # print(importance_attention_head[0])
    exit(1)




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

# load model checkpoints
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tell pytorch to run the model on GPU
model.cuda()

se = senteval.engine.SE(params_senteval, batcher, prepare)

# the 7 commonly used semantic textual similarity (STS) datasets used by the authors 
# sts_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
sts_tasks = 'STS12'

# creates an object of the class associated with the sts_task, which loads the required datasets,
# then initializes the created object's similarity field and runs the prepare function.
# finally, runs the batcher function to get the sentence-level encodings for each batch and computes the average
# pearson and spearman scores for the dataset
results = se.eval(sts_tasks)
print(results)