from first_last_gridsearch import grid_search
from transformers import AutoModel, AutoTokenizer

# model and tokenizer to used for the grid search
# note: moved outside of first_last_gridsearch.py to eliminate the need to download the model and tokenizer
# for every call to grid_search. this is made possible by the fact that ditto is a learning-free method
# aka the weights of the pretrained model are not modified.

# for testing
# used to confirm grid_search method was similar to the authors
# model = AutoModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# load model checkpoints for multilingual model
model = AutoModel.from_pretrained("bert-base-multilingual-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

# constants corresponding to the number of attention layers and attention heads per layer for desired model
NUM_ATTENTION_LAYERS = 12
NUM_ATTENTION_HEADS = 12

# used to keep track of the best dev score and corresponding attention layer/attention head
best_dev_score = 0
best_attention_layer = -1
best_attention_head = -1 

multilingual_stsb_tasks = ['STSBenchmarkES_gs', 'STSBenchmarkFR_gs', 'STSBenchmarkIT_gs', 'STSBenchmarkPT_gs']

# for testing
# multilingual_stsb_tasks = ['STSBenchmark_gs']
results = {}

for ml_task in multilingual_stsb_tasks:
    # perform grid search over all the BERT/mBERT attention layers and attention heads of those layers
    for i in range(1, NUM_ATTENTION_LAYERS + 1):
        for j in range(1, NUM_ATTENTION_HEADS + 1):
            temp = grid_search((i, j), model, tokenizer, ml_task)
            if (temp > best_dev_score):
                best_dev_score = temp
                best_attention_layer = i
                best_attention_head = j
    
    # keep track of best dev score, best attention layer, and best attention head for each language
    results[ml_task] = {'best_dev_score' : best_dev_score, 'best_attention_layer' : best_attention_layer, 
                        'best_attention_head' : best_attention_head}
    
    best_dev_score = 0
    best_attention_layer = -1
    best_attention_head = -1

# print out grid search results
for ml_task in multilingual_stsb_tasks:
    print(f'{ml_task}')
    print(f'The Attention Layer with the best STSBenchmark Dev Score was {results[ml_task]["best_attention_layer"]}')
    print(f'The Attention Head with the best STSBenchmark Dev Score was {results[ml_task]["best_attention_head"]}')
    print(f'The Best STSBenchmark Dev Score was {results[ml_task]["best_dev_score"]}')
    print()