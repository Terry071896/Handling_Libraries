import torch
import pandas as pd

from torch import tensor 
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


from datasets import load_dataset, load_metric

from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr import visualization as viz

import matplotlib.pyplot as plt

import argparse 
import jsonlines
import os 

from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# model_path = "/scratch/general/vast/u1427155/cs6966/assignment1/models/microsoft/deberta-v3-base-finetuned-imdb/checkpoint-12500"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)

def predict(inputs):
    outputs = model(torch.tensor([inputs]))
    return output.start_logits, output.end_logits, output.attentions


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = model(inputs_embeds=inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )
    pred = pred[position]
    return pred.max(1).values

ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence


idx=0
failed = []
import json

with open('../data/failed.jsonl', 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    failed.append(result)
    tokens = tokenizer(result['review'], truncation=True)

    


