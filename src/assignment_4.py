
'''
Code source (with some changes):
https://levelup.gitconnected.com/huggingface-transformers-interpretability-with-captum-28e4ff4df234
https://gist.githubusercontent.com/theDestI/fe9ea0d89386cf00a12e60dd346f2109/raw/15c992f43ddecb0f0f857cea9f61cd22d59393ab/explain.py
'''

import torch
import pandas as pd
import numpy as np

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

from torchtext.vocab import Vocab

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset

if torch.__version__ >= '1.7.0':
    norm_fn = torch.linalg.norm
else:
    norm_fn = torch.norm

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
    
    def forward_func(self, inputs: tensor, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                       attention_mask=torch.ones_like(inputs))
        return pred[position]
        
    def visualize(self, scores_mat, outfile_path: str, x_label_name='Head'):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        
        fig = plt.figure(figsize=(100, 100))

        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(4, 3, idx+1)
            # append the attention weights
            im = ax.imshow(scores, cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(self.all_tokens)))
            ax.set_yticks(range(len(self.all_tokens)))

            ax.set_xticklabels(self.all_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(self.all_tokens, fontdict=fontdict)
            ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(outfile_path)
                      
    def explain(self, text: str, outfile_path: str):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method.
        """
        #prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text=text)
        outputs = self.__pipeline.model(inputs, attention_mask=torch.ones_like(inputs), output_attentions=True)
        output_attentions = outputs.attentions
        output_attentions_all = torch.stack(output_attentions)  
        indices = inputs[0].detach().tolist()
        self.all_tokens = self.__pipeline.tokenizer.convert_ids_to_tokens(indices)
        
        self.visualize(norm_fn(output_attentions_all, dim=2).squeeze().detach().cpu().numpy(), outfile_path=outfile_path)
        # inputs = self.generate_inputs(text)
        # baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        
        # lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, 'deberta').embeddings)
        
        # attributes, delta = lig.attribute(inputs=inputs,
        #                           baselines=baseline,
        #                           target = self.__pipeline.model.config.label2id[prediction[0]['label']], 
        #                           return_convergence_delta = True)
        # Give a path to save
        #self.visualize(inputs, attributes, outfile_path)
    
    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False), device = self.__device).unsqueeze(0)
    
    def generate_baseline(self, sequence_len: int) -> tensor:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """        
        return torch.tensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id], device = self.__device).unsqueeze(0)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint) 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels)
    dataset = load_dataset('imdb')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                device=device
                                )
    exp_model = ExplainableTransformerPipeline(args.model_checkpoint, clf, device)

    

    idx=0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            exp_model.explain(obj["review"], os.path.join(args.output_dir,f'example_{idx}'))
            idx+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analsis_dir', default='out', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    parser.add_argument('--a1_analysis_file', type=str, default='out/a1_analysis_data.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='out', type=str, help='Directory where model checkpoints will be saved')    
    args = parser.parse_args()
    main(args)