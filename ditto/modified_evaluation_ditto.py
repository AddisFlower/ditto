# Copyright (c) 2023, Alibaba Group
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import json

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Function that prints the results in a more readable format
def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main():
    # Parses through the arguments that were passed to the program and set the parameters accordingly
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str,
                        choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last',
                                 'att_first_last', 'tfidf_first_last', 'avg_static', 'att_last', 'att_static',
                                 ],
                        default='cls',
                        help="Which pooler to use")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+',
                        default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'SICKRelatedness', 'STSBenchmark'],
                        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    parser.add_argument("--layer", default=0, type=int)
    parser.add_argument("--head", default=0, type=int)

    args = parser.parse_args()

    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Open the json file that contains the dictionary that has average TF-IDF weight for each token encountered in the Wikipedia file 
    with open('average_tfidf.json', 'r') as json_file:
        average_tfidf = json.load(json_file)
    
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Start of code added by Nathan 
        sentence_importance_list = [] # A list that contains a list for each sentence in the batch. Each of these lists contains the TF-IDF weights of each encoded token.
        for i in range(len(batch['input_ids'])): # For each sentence in the batch
            token_importance_list = [] # List that will contain the TF-IDF weights of each encoded token in that sentence

            token_ids = batch['input_ids'][i]  # Get the token IDs
            tokens = tokenizer.convert_ids_to_tokens(token_ids) # Convert the token IDs back to tokens
            for token in tokens: # For each encoded token in the sentence
                if token in average_tfidf:
                    token_importance_list.append(average_tfidf[token]) # If the given token is found in the dictionary, add the TF-IDF weight of that token to the token_importance_list
                else:
                    token_importance_list.append(0) # If the given token is not found in the dictionary, add 0 to the token_importance_list. This is important because it weighs the padding tokens by 0.
            sentence_importance_list.append(token_importance_list)
        # End of code added by Nathan


        for k in batch:
            batch[k] = batch[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, output_attentions=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            if hasattr(outputs, 'pooler_output'):
                pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
            attention = outputs.attentions
            attention_diag = torch.diagonal(attention[args.layer][:, args.head, :, :], 0, dim1=1, dim2=2)
        

        if args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()
        
        # Start of code added by Nathan
        # Implementation of BERT first-last TF-IDF pooling 
        elif args.pooler == 'tfidf_first_last':
            first_hidden = hidden_states[0] # Get the first hidden state
            last_hidden = hidden_states[-1] # Get the last hidden state
            hiddens = first_hidden + last_hidden # Add the first and last hidden state
            for i in range(len(hiddens)): # Going thorugh each sentence in the batch
                for j in range(len(hiddens[i])): # Going through each token in the sentence
                    hiddens[i][j] = hiddens[i][j] * sentence_importance_list[i][j] # Multiplying the TF-IDF weight of the token with its corresponding hidden state vector
            pooled_result = ((hiddens).cpu() / 2.0 * batch['attention_mask'].unsqueeze(
                -1).cpu()).sum(1) # Take the average like instructed in the paper
            return pooled_result.cpu()
        # End of code added by Nathan 

        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(
                -1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch[
                'attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / \
                            batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "att_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(
                -1) * attention_diag.unsqueeze(-1)).sum(1)
            return pooled_result.cpu()
        elif args.pooler == "att_last":
            pooled_result = (last_hidden * batch['attention_mask'].unsqueeze(-1) * attention_diag.unsqueeze(-1)).sum(1)
            return pooled_result.cpu()
        elif args.pooler == "att_static":
            with torch.no_grad():
                last_hidden = model.embeddings.word_embeddings(batch['input_ids'])
            pooled_result = (last_hidden * batch['attention_mask'].unsqueeze(-1) * attention_diag.unsqueeze(-1)).sum(1)
            return pooled_result.cpu()
        elif args.pooler == "avg_static":
            with torch.no_grad():
                last_hidden = model.embeddings.word_embeddings(batch['input_ids'])
            pooled_result = (last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(
                -1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError
    


    results = {}


    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
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


if __name__ == "__main__":
    main()
