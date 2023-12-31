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
from captum.attr import LayerIntegratedGradients, TokenReferenceBase

import matplotlib.pyplot as plt
import re
import argparse 
import jsonlines
import os 

from util import *

"""
Average IOU F1 Score: 0.19651805975062117, standard deviation: 0.10325654153274418
Average Token F1 Score: 0.19651805975062117, standard deviation: 0.10325654153274416
Average comprehensiveness: 0.0015938799632223028, standard deviation: 0.002006300500907498
Average sufficiency: 0.0, standard deviation: 0.0
"""


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
        
    def visualize(self, inputs: list, attributes: list, outfile_path: str):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        #import pdb; pdb.set_trace()
        attr_sum = attributes.sum(-1) 
        
        attr = attr_sum / torch.norm(attr_sum)
        
        a = pd.Series(attr.cpu().numpy()[0][::-1], 
                         index = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])[::-1])
        a.plot.barh(figsize=(10,20))
        plt.savefig(outfile_path)
    
    def return_highlight(self, inputs: list, attributes: list, proportion: float):
        attr_sum = attributes.sum(-1) 
        
        attr = attr_sum / torch.norm(attr_sum)
        
        k = round(proportion * len(attr.cpu().numpy()[0][::-1]))
        index = np.argsort(np.absolute(attr.cpu().numpy()[0][::-1]))[:k]
        return np.array(self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])[::-1])[index].tolist()
                      
    def explain(self, text: str, outfile_path: str, proportion: float):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method. 
        """
        prediction = self.predict_text(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        
        lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, 'deberta').embeddings)
        
        attributes, delta = lig.attribute(inputs=inputs,
                                  baselines=baseline,
                                  target = self.__pipeline.model.config.label2id[prediction[0]['label']], 
                                  return_convergence_delta = True)
        # Give a path to save
        return self.return_highlight(inputs, attributes, proportion)
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
    
    def predict_text(self, text: str) -> np.ndarray:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """        
        return self.__pipeline.predict(text)
    

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, cache_dir = args.cache_dir) 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels, cache_dir = args.cache_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                device=device
                                )
    exp_model = ExplainableTransformerPipeline(args.model_checkpoint, clf, device)

    idx=0
    val_dataset = []
    
    with jsonlines.open(os.path.join(args.analsis_dir, args.analysis_file), 'r') as reader:
        for obj in reader:
            val_dataset.append(obj)


    iou_f1_scores = []
    token_f1_scores = []
    comprehensiveness_list = []
    sufficiency_list = []

    for i, obj in enumerate(val_dataset):
        print("The length of the review: {}.".format(len(obj["input"])))
        if len(obj["input"])<=3900: # avoiding cuda out of memory
            len_highlight = 0
            for highlight in obj["highlight"]:
                len_highlight += len(highlight)
            #proportion = len_highlight/len(obj["input"])
            raw_highlight = exp_model.explain(obj["input"], os.path.join(args.output_dir,f'example_{idx}'),len_highlight)
            #print("raw_highlight", raw_highlight)
            cleaned_highlight = [token.replace('▁', '') for token in raw_highlight if re.search('[a-zA-Z]', token)]
            #print("cleaned_highlight", cleaned_highlight)

            final_highlight = []
            current_len_highlight = 0
            for token in cleaned_highlight:
                if current_len_highlight + len(token) > len_highlight:
                    break
                final_highlight.append(token)
                current_len_highlight += len(token)

            #print("final_highlight",final_highlight)

            #highlight = exp_model.explain(obj["input"], os.path.join(args.output_dir,f'example_{idx}'), proportion)
            #print("highlight",highlight)

            val_dataset[i]['gradient_highlight'] = final_highlight

            target_labels = [word for label in obj['highlight'] for word in label.split() if word.isalpha()]

            print(str(final_highlight), str(target_labels))

            '''
            iou_f1 = iou_f1_score(str(final_highlight), str(target_labels))
            val_dataset[i]['gradient_based_iou_f1_score'] = iou_f1
            token_f1 = token_f1_score(str(final_highlight), str(target_labels))
            val_dataset[i]['gradient_based_token_f1_score'] = token_f1            
            '''
            iou_f1 = iou_f1_score(' '.join(final_highlight), ' '.join(target_labels))
            val_dataset[i]['gradient_based_iou_f1_score'] = iou_f1
            token_f1 = token_f1_score(' '.join(final_highlight).replace("_", ""), ' '.join(target_labels))
            val_dataset[i]['gradient_based_token_f1_score'] = token_f1
            
            print("iou_f1",iou_f1)
            print("token_f1",token_f1)

            iou_f1_scores.append(iou_f1)
            token_f1_scores.append(token_f1)

            ##=====================================## 
            ##for comprehensiveness and sufficiency##
            full_input_prediction = exp_model.predict_text(obj["input"])[0]['score']
            print(f"full_input_prediction: {full_input_prediction}")

            val_input_for_comprehensiveness = obj['input']
            val_input_for_sufficiency = ""
            for highlight in final_highlight:
                val_input_for_comprehensiveness = val_input_for_comprehensiveness.replace(highlight, "   ")
                val_input_for_sufficiency += (highlight + "   ")
        
            print("val_input_for_comprehensiveness: ", val_input_for_comprehensiveness)
            print("val_input_for_sufficiency: ", val_input_for_sufficiency)

            comprehensiveness_input_prediction = exp_model.predict_text(val_input_for_comprehensiveness)[0]['score']
            sufficiency_input_prediction = exp_model.predict_text(val_input_for_sufficiency)[0]['score']
            print(f"comprehensiveness_input_prediction: {comprehensiveness_input_prediction}")
            print(f"sufficiency_input_prediction: {sufficiency_input_prediction}")

            comprehensiveness = full_input_prediction - comprehensiveness_input_prediction if full_input_prediction > comprehensiveness_input_prediction else 0
            sufficiency = full_input_prediction - sufficiency_input_prediction if full_input_prediction > sufficiency_input_prediction else 0
            
            comprehensiveness_list.append(comprehensiveness)
            sufficiency_list.append(sufficiency)
            val_dataset[i]['comprehensiveness'] = comprehensiveness
            val_dataset[i]['sufficiency'] = sufficiency

        idx+=1
        print (f"Example {idx} done")

    from numpy import mean, std
    print(f"Average IOU F1 Score: {mean(iou_f1_scores) }, standard deviation: {std(iou_f1_scores)}")
    print(f"Average Token F1 Score: {mean(token_f1_scores)}, standard deviation: {std(token_f1_scores)}")
    print(f"Average comprehensiveness: {mean(comprehensiveness_list) }, standard deviation: {std(comprehensiveness_list)}")
    print(f"Average sufficiency: {mean(sufficiency_list)}, standard deviation: {std(sufficiency_list)}")

    out = os.path.join(args.output_dir, 'gradient_based_' + args.analysis_file)
    with jsonlines.open(out, mode='w') as writer:
        for item in val_dataset:
            writer.write(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default = 123, type=int, help='Random seed set') # <======================> 
    parser.add_argument('--analsis_dir', default='out', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    #parser.add_argument('--model_checkpoint', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='model checkpoint') #This is only for text generation
    parser.add_argument('--analysis_file', type=str, default='val.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='out', type=str, help='Directory where model checkpoints will be saved')    
    parser.add_argument('--cache_dir', type=str, help='Directory where cache will be saved')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)