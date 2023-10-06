import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch import tensor 
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr import visualization as viz

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerActivation
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import sys
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/scratch/general/vast/u1427155/cs6966/assignment1/models/microsoft/deberta-v3-base-finetuned-imdb/checkpoint-12500"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)

clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                device=device
                                )

def predict(inputs, token_type_ids, position_ids, attention_mask):
    #outputs = model(torch.tensor([inputs]), token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask, )
    outputs = model(torch.tensor([inputs]))
    predicted_label = outputs.logits.argmax(-1)
    return predicted_label, outputs.attentions

ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence


def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

# interpretable_embedding = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')

# def construct_whole_bert_embeddings(input_ids, ref_input_ids, \
#                                     token_type_ids=None, ref_token_type_ids=None, \
#                                     position_ids=None, ref_position_ids=None):
#     input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
#     ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)
    
#     return input_embeddings, ref_input_embeddings

def visualize_token2token_scores(scores_mat, x_label_name='Head'):
    fig = plt.figure(figsize=(100, 100))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(4, 3, idx+1)
        # append the attention weights
        im = ax.imshow(scores, cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(all_tokens)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(all_tokens, fontdict=fontdict)
        ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(script_directory, '..out/'+x_label_name.split(':')[0]+'layers.png'))
    #plt.show() 

def visualize_token2head_scores(scores_mat):
    fig = plt.figure(figsize=(30, 50))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(6, 2, idx+1)
        # append the attention weights
        im = ax.matshow(scores_np, cmap='viridis')

        fontdict = {'fontsize': 20}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(scores)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(range(len(scores[0])), fontdict=fontdict)
        ax.set_xlabel('Layer {}'.format(idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / norm_fn(attributions)
    return attributions

if torch.__version__ >= '1.7.0':
    norm_fn = torch.linalg.norm
else:
    norm_fn = torch.norm

idx=0
failed = []
failed_tokens = []
import json

with open(os.path.join(script_directory, '../data/failed.jsonl'), 'r') as json_file:
    json_list = list(json_file)

for i, json_str in enumerate(json_list):
    result = json.loads(json_str)
    failed.append(result)
    tokens = tokenizer(result['review'], truncation=True)
    input_ids = tokens['input_ids']
    failed_tokens.append(input_ids)
    
    input_ids, ref_input_ids = construct_input_ref_pair(result['review'], ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, -1)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    pred_class, output_attentions = predict(input_ids,
                                   token_type_ids=token_type_ids, \
                                   position_ids=position_ids, \
                                   attention_mask=attention_mask)

    # shape -> layer x batch x head x seq_len x seq_len
    output_attentions_all = torch.stack(output_attentions)  
    visualize_token2token_scores(norm_fn(output_attentions_all, dim=2).squeeze().detach().cpu().numpy(),
                             x_label_name='Example_%s: Layer'%i) 

    # layer_attrs_start = []
    # layer_attrs_end = []

    # layer_attn_mat_start = []
    # layer_attn_mat_end = []

    # input_embeddings, ref_input_embeddings = construct_whole_bert_embeddings(input_ids, ref_input_ids, \
    #                                         token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids, \
    #                                         position_ids=position_ids, ref_position_ids=ref_position_ids)

    # for i in range(model.config.num_hidden_layers):
    #     lc = LayerConductance(squad_pos_forward_func, model.bert.encoder.layer[i])
    #     layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(token_type_ids, position_ids,attention_mask, 0))
    #     layer_attributions_end = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings, additional_forward_args=(token_type_ids, position_ids,attention_mask, 1))
        
    #     layer_attrs_start.append(summarize_attributions(layer_attributions_start[0]))
    #     layer_attrs_end.append(summarize_attributions(layer_attributions_end[0]))

    #     layer_attn_mat_start.append(layer_attributions_start[1])
    #     layer_attn_mat_end.append(layer_attributions_end[1])

    

    

