import argparse
import numpy as np
import torch
import jsonlines 
import random 
import os
from sklearn.metrics import f1_score

def iou_f1_score(generated_explanation, target_explanation):
    # Convert explanations to sets of words
    generated_set = set(generated_explanation.split())
    target_set = set(target_explanation.split())

    # Calculate the intersection and union of the sets
    intersection = len(generated_set.intersection(target_set))
    union = len(generated_set) + len(target_set) - intersection

    # Calculate IOU F1 score
    if union == 0:
        return 1.0  # Handle the case of an empty union
    else:
        iou = intersection / union
        f1 = 2 * (iou) / (iou + 1)  # Calculate F1 score from IOU
        return f1
    

def newyorker_caption_contest_data(args):
    from datasets import load_dataset
    dset = load_dataset(args.task_name, cache_dir = args.cache_dir)

    res = {}
    for spl, spl_name in zip([dset['train'], dset['test']],
                            ['train', 'test']):
        cur_spl = []
        for inst in list(spl):
            inp = inst['text']
            targ = inst['label']
            cur_spl.append({'input': inp, 'target': targ})
        
            #'input' is an image annotation we will use for a llama2 e.g. "scene: the living room description: A man and a woman are sitting on a couch. They are surrounded by numerous monkeys. uncanny: Monkeys are found in jungles or zoos, not in houses. entities: Monkey, Amazon_rainforest, Amazon_(company)."
            #'target': a human-written explanation 
            #'image': a PIL Image object
            #'caption_choices': is human-written explanation

        res[spl_name] = cur_spl
    return res


def newyorker_caption_contest_llama2(args): 
    print("Loading data")
    #nyc_data = newyorker_caption_contest_data(args)
    #nyc_data_five_val = random.sample(nyc_data['test'],5)
    #nyc_data_train_two = random.sample(nyc_data['train'],2)
    
    print ("Loading data")
    nyc_data_five_val = []
    with jsonlines.open('out/val.jsonl') as reader:
        for obj in reader:
            nyc_data_five_val.append(obj)

    nyc_data_train_two = []
    with jsonlines.open('out/train.jsonl') as reader:
        for obj in reader:
            nyc_data_train_two.append(obj)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("Loading model")
    '''
    Ideally, we'd do something similar to what we have been doing before: 

        tokenizer = AutoTokenizer.from_pretrained(args.llama2_checkpoint, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(args.llama2_checkpoint, torch_dtype=torch.float16, device_map="auto")
        tokenizer.pad_token = tokenizer.unk_token_id
        
        prompts = [ "our prompt" for val_inst in nyc_data_five_val]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        output_sequences = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_text = [tokenizer.decode(s, skip_special_tokens=True) for s in output_sequences]

    But I cannot produce text with this prototypical code with HF llama2. 
    Thus we will use pipeline instead. 
    '''
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llama2_checkpoint, cache_dir = args.cache_dir)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.llama2_checkpoint,
        #cache_dir = args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    for i, val_inst in enumerate(nyc_data_five_val):         
        # ======================> ADD YOUR CODE TO DEFINE A PROMPT WITH TWO TRAIN EXAMPLES/DEMONSTRATIONS/SHOTS <======================
        prompt =  "Please use yes or not to answer whether the movie review is positive or not and then use report important phrases to explain the reason?" + \
            nyc_data_train_two[0]['input'] + "[/INST]" + nyc_data_train_two[0]['target'] + "</s><s>[INST]" + \
                "Please use yes or not to answer whether the movie review is positive or not and then use report important phrases to explain the reason?" +\
                      nyc_data_train_two[1]['input'] + "[/INST]"  + nyc_data_train_two[1]['target'] + "</s><s>[INST]" + \
                        "According to the above two examples, please use yes or not to answer whether the movie review is positive or not and then use report important phrases to explain the reason?" + \
                              val_inst['input'] + "[/INST]"         

        sequences = pipeline(
            prompt,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1024,
        )
        
        gen_expl = sequences[0]['generated_text'].split("/INST]")[-1]
        nyc_data_five_val[i]['generated_llama2']=gen_expl

        #calculate Intersection-over-Union (IOU) F1
        target_explanation = nyc_data_five_val[i]['target']
        iou_f1 = iou_f1_score(gen_expl, target_explanation)
        nyc_data_five_val[i]['iou_f1_score'] = iou_f1


    filename = 'out/val.jsonl'
    with jsonlines.open(filename, mode='w') as writer:
        for item in nyc_data_five_val:
            writer.write(item)

    filename = 'out/train.jsonl'
    with jsonlines.open(filename, mode='w') as writer:
        for item in nyc_data_train_two:
            writer.write(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default = 1418772, type=int, help='Random seed set to your uNID') # <======================> 
    parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
    parser.add_argument('--cache_dir', type=str, help='Directory where cache will be saved')
    parser.add_argument('--task_name', default="imdb",  type=str, help='Name of the task that will be used by huggingface load dataset')    
    #parser.add_argument('--subtask', default="explanation", type=str, help="The contest has three subtasks: matching, ranking, explanation")
    parser.add_argument('--llama2_checkpoint', default="meta-llama/Llama-2-7b-chat-hf", type=str, help="The hf name of a llama2 checkpoint")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    newyorker_caption_contest_llama2(args)