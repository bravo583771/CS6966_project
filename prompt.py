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
    

def newyorker_caption_contest_data(task_name = 'movie_rationales'):
    from datasets import load_dataset
    dset = load_dataset(task_name)

    res = {}
    for spl, spl_name in zip([dset['train'], dset['validation'], dset['test']],
                            ['train', 'val', 'test']):
        cur_spl = []
        for inst in list(spl):
            inp = inst['review']
            targ = inst['label']
            highlight = inst['evidences']
            cur_spl.append({'input': inp, 'target': targ, 'highlight': highlight})
        
            #'review' is an review we will use for a llama2 e.g. "Some TV programs continue into embarrassment (my beloved 'X-Files' comes to mind.) I've been a fan of Dennis Farina since 'Crime Story,' another late, lamented show. 'Buddy Faro' never had a chance. The series had a good premise and great actors. It's really, really a shame."
            #'label': a label
            #'evidences': a human-written important phrases

        res[spl_name] = cur_spl
    return res

def label4prompt(label):
    # 0 = negative, 1 = positive
    return "yes" if label==1 else "no" 
    #return "Yes, the movie review is positive." if label==1 else "No, the movie review is negative." 
    #return "positive" if label==1 else "negative" 
    #return "The movie review is " + "positive." if label==1 else "negative." 

def newyorker_caption_contest_llama2(args): 
    print("Loading data")
    nyc_data = newyorker_caption_contest_data(args.task_name)
    nyc_data_five_val = random.sample(nyc_data['val'],5)
    nyc_data_train_two = random.sample(nyc_data['train'],2)
    
    #print ("Loading data")
    #nyc_data_five_val = []
    #with jsonlines.open('out/val.jsonl') as reader:
    #    for obj in reader:
    #        nyc_data_five_val.append(obj)

    #nyc_data_train_two = []
    #with jsonlines.open('out/train.jsonl') as reader:
    #    for obj in reader:
    #        nyc_data_train_two.append(obj)

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
        label_train_0 = label4prompt(nyc_data_train_two[0]['target']) # 0 = negative, 1 = positive
        label_train_1 = label4prompt(nyc_data_train_two[1]['target']) # 0 = negative, 1 = positive
        #prompt =  "Please use yes or not to answer whether the movie review is positive or not and then use report important phrases to explain the reason?" +\
        #    nyc_data_train_two[0]['input'] + "[/INST]" + label_train_0 +\
        #        " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
        #            "Please use yes or not to answer whether the movie review is positive or not and then use report important phrases to explain the reason?" +\
        #              nyc_data_train_two[1]['input'] + "[/INST]"  + label_train_1 +\
        #                 " the important phrases are " + str(nyc_data_train_two[1]['highlight']) + "</s><s>[INST]" +\
        #                    "According to the above two examples, please use yes or not to answer whether the movie review is positive or not and then use report important phrases to explain the reason?" +\
        #                      val_inst['input'] + "[/INST]"     
        prompt =  "Please use yes or not to answer whether the movie review is positive or negative and then report important phrases to explain the reason." +\
            nyc_data_train_two[0]['input'] + "[/INST]" + label_train_0 +\
                " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                    "According to the above example, please use yes or not to answer whether the movie review is positive or negative and then report important phrases to explain the reason." +\
                        val_inst['input'] + "[/INST]"    

        sequences = pipeline(
            prompt,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4096,
        )
        
        gen_expl = sequences[0]['generated_text'].split("/INST]")[-1]
        nyc_data_five_val[i]['generated_llama2']=gen_expl

        #calculate Intersection-over-Union (IOU) F1
        label_val = label4prompt(nyc_data_five_val[i]['target']) # 0 = negative, 1 = positive
        target_explanation = label_val + " the important phrases are " + str(nyc_data_five_val[i]['highlight'])
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
    parser.add_argument('--task_name', default="movie_rationales",  type=str, help='Name of the task that will be used by huggingface load dataset')    
    #parser.add_argument('--subtask', default="explanation", type=str, help="The contest has three subtasks: matching, ranking, explanation")
    parser.add_argument('--llama2_checkpoint', default="meta-llama/Llama-2-7b-chat-hf", type=str, help="The hf name of a llama2 checkpoint")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    newyorker_caption_contest_llama2(args)