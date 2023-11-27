import argparse
import numpy as np
import torch
import jsonlines 
import random 
import re
#from sklearn.metrics import f1_score
import torch.nn.functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from util import *


device = "cuda" if torch.cuda.is_available() else "cpu"

def movie_rationales_llama2(args): 
    print("Loading data")
    val_filename = 'out/val.jsonl'
    train_filename = 'out/train.jsonl'
    if os.path.isfile(val_filename):
        nyc_data_train_two = []
        nyc_data_val = []
        with jsonlines.open(val_filename) as reader:
            for obj in reader:
                nyc_data_val.append(obj)
        with jsonlines.open(train_filename) as reader:
            for obj in reader:
                nyc_data_train_two.append(obj)
    else:
        nyc_data = movie_rationales_data(args.task_name)
        nyc_data_train_two = random.sample(nyc_data['train'], 2)
        nyc_data_val = random.sample(nyc_data['val'], 200)
        with jsonlines.open(train_filename, mode='w') as writer:
            for item in nyc_data_train_two:
                writer.write(item)
        with jsonlines.open(val_filename, mode='w') as writer:
            for item in nyc_data_val:
                if len(item["input"])<=3900:
                    writer.write(item)
                else:
                    nyc_data_val.remove(item)

    print(f"Size of Validation after removing too long input: {len(nyc_data_val)}")

    print("Loading model")
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    access_token = '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'
    tokenizer = AutoTokenizer.from_pretrained(args.llama2_checkpoint, cache_dir = args.cache_dir, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(args.llama2_checkpoint, output_scores=True, return_dict_in_generate=True, token=access_token).to(device)

    iou_f1_scores = []
    token_f1_scores = []
    comprehensiveness_list = []
    sufficiency_list = []

    ##=====================================## 
    ##For Faithfulness##
    ##for comprehensiveness and sufficiency##
    input0_for_comprehensiveness = nyc_data_train_two[0]['input']
    input0_for_sufficiency = "   "
    for highlight in nyc_data_train_two[0]['highlight']:
        input0_for_comprehensiveness = input0_for_comprehensiveness.replace(highlight, "  ")
        input0_for_sufficiency += (highlight + "   ")

    input1_for_comprehensiveness = nyc_data_train_two[1]['input']
    input1_for_sufficiency = "   "
    for highlight in nyc_data_train_two[1]['highlight']:
        input0_for_comprehensiveness = input1_for_comprehensiveness.replace(highlight, "  ")
        input1_for_sufficiency += (highlight + "   ")
    ##=====================================## 

    
    start_item = 0
    """
    #This is for continous running in different jobs
    filename = 'out/val_' + args.prompt + '.jsonl'
    
    if os.path.isfile(filename):
        with jsonlines.open(filename) as reader:
            for i, item in enumerate(reader):
                if nyc_data_val[i]["input"]==item["input"] and len(set(['comprehensiveness', 'sufficiency', 'iou_f1_score', 'token_f1_score']).intersection(item.keys()))==4:
                    nyc_data_val[i] = item
                    start_item = i+1
                    iou_f1_scores.append(item['iou_f1_score'])
                    token_f1_scores.append(item['token_f1_score'])
                    comprehensiveness_list.append(item['comprehensiveness'])
                    sufficiency_list.append(item['sufficiency'])
                else:
                    break
    """
    count = 0 
    for i, val_inst in enumerate(nyc_data_val):       
        if i < start_item:
            continue
        else:
            if count >= args.val_size:
                break
            count += 1  
        # ======================> ADD YOUR CODE TO DEFINE A PROMPT WITH TWO TRAIN EXAMPLES/DEMONSTRATIONS/SHOTS <======================
        label_train_0 = label4prompt(nyc_data_train_two[0]['target']) # 0 = negative, 1 = positive
        label_train_1 = label4prompt(nyc_data_train_two[1]['target']) # 0 = negative, 1 = positive
        

        if args.prompt == "two_shot":
            '''
            #1: 0.21913117277650862 for 5 val instances
            
            prompt =  "<s>[INST] Please use yes or no to answer whether the movie review is positive or not and then list important phrases" +\
                nyc_data_train_two[0]['input'] + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "Please use yes or no to answer whether the movie review is positive or not and then list important phrases" +\
                          nyc_data_train_two[1]['input'] + "[/INST]"  + label_train_1 +\
                            " the important phrases are " + str(nyc_data_train_two[1]['highlight']) + "</s><s>[INST]" +\
                                "According to the above two examples, please use yes or no to answer whether the movie review is positive or not and then list important phrases" +\
                                    val_inst['input'] + "[/INST]"           
            '''
            
            #Average IOU F1 Score: 0.24472592482987424 for 200 instances
            #Average Token F1 Score: 0.24472592482987424 for 200 instances
            prompt =  "<s>[INST] Please answer whether the movie review is positive or negative and then list the important phrases." +\
                nyc_data_train_two[0]['input'] + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "Please answer whether the movie review is positive or negative and then list the important phrases." +\
                          nyc_data_train_two[1]['input'] + "[/INST]"  + label_train_1 +\
                            " the important phrases are " + str(nyc_data_train_two[1]['highlight']) + "</s><s>[INST]" +\
                                "please answer whether the movie review is positive or negative and then list the important phrases. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                                    val_inst['input'] + "[/INST]"              
                     
        elif args.prompt == "one_shot":

            
            # 1: 0.194 for 5 val instances
            #Average IOU F1 Score: 0.23042911440679834 for 200 instances
            #Average Token F1 Score: 0.2304291144067983 for 200 instances
            prompt =  "<s>[INST] Please answer whether the movie review is positive or negative and then report important phrases to explain the reason." +\
                nyc_data_train_two[0]['input'] + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "please answer whether the movie review is positive or negative and then list the important phrases. Format the response starting with either 'the review is positive' or 'the review is negative. Again, the response should start with either 'the review is positive' or 'the review is negative." +\
                            val_inst['input'] + "[/INST]"                
        
            '''
            #2: 0.21088923922994768 for 5 val instances
            #Even though the iou score is higher, the answer is incorrect 
            #Yes, the movie is negative -----should be ----->  No, the movie is nagative
            prompt =  "<s>[INST] Please use yes or no to answer whether the movie review is positive or negative and then list important phrases" +\
                nyc_data_train_two[0]['input'] + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "According to the above example, please use yes or no to answer whether the movie review is positive or negative and then list important phrases" +\
                            val_inst['input'] + "[/INST]"            
            '''
        elif args.prompt == "zero_shot":
            
            #0.19329531802908942 for 5 val instances
            #prompt =  "<s>[INST] Please use yes or no to answer whether the movie review is positive or not and then list important phrases" +\
            #                val_inst['input'] + "[/INST]" 

            #Average IOU F1 Score: 0.23420844552028455 for 200 instances
            #Average Token F1 Score: 0.23420844552028455 for 200 instances

            prompt =  "<s>[INST] Please answer whether the movie review is positive or negative and then list the important phrases. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                            val_inst['input'] + "[/INST]"  
            
        """
        max_token_limit = 4096
        
        sequences = pipeline(  # this will be the class "transformers.pipelines.text_generation"
            prompt,
            #return_tensors = True,
            return_text = True,
            #output_scores=True, 
            # according to the source code, 
            # seems like llama2 does not support returning score or probability values in "transformers.pipelines.text_generation" 
            # https://huggingface.co/transformers/v4.4.2/_modules/transformers/pipelines/text_generation.html
            do_sample = True,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_token_limit,
        )

        """

        input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(device)

        # Generate text output
        model_output = model.generate(
            input_ids,
            max_length=4096,
            num_return_sequences=1, 
            output_scores=True, 
            return_dict_in_generate=True,
            do_sample=True,
            #eos_token_id=tokenizer.eos_token_id,
        )
        # Decode the generated token IDs to text
        #generated_text = tokenizer.decode(model_output[0], skip_special_tokens=True).split("/INST]")[-1]
        generated_text = tokenizer.decode(model_output[0][0], skip_special_tokens=True).split("/INST]")[-1]
        generated_text = generated_text.strip()
        print("generated_text",generated_text)
        print("--------------")

        ####extract important phrases######
        data = []
        pattern = r'\"([^\"]+)\"'

        explanation_lines = generated_text.split('\n')
        #print("Full Explanation:", explanation_lines)
        #print("---------------------------")    

        important_phrases = []
        for line in explanation_lines:
            matches = re.findall(pattern, line)
            important_phrases += matches

        #print("Important Phrases:", important_phrases)
        #print("---------------------------")

        data.append(important_phrases)     

        ##Plausibility##
        #calculate Intersection-over-Union (IOU) F1 anf Token F1
        label_val = label4prompt(nyc_data_val[i]['target']) # 0 = negative, 1 = positive
        #target_explanation = label_val + " the important phrases are " + str(nyc_data_val[i]['highlight'])
        target_explanation = label_val + " the important phrases are " + ' '.join(nyc_data_val[i]['highlight'])
        iou_f1 = iou_f1_score(generated_text, target_explanation)
        nyc_data_val[i]['iou_f1_score'] = iou_f1
        token_f1 = token_f1_score(generated_text, target_explanation)
        nyc_data_val[i]['token_f1_score'] = token_f1

        iou_f1_scores.append(iou_f1)
        token_f1_scores.append(token_f1)
        #print("Instance {} done.".format(i))


        ##=====================================## 
        ##for comprehensiveness and sufficiency##
        val_input_for_comprehensiveness = val_inst['input']
        val_input_for_sufficiency = ""
        for highlight in important_phrases:
            val_input_for_comprehensiveness = val_input_for_comprehensiveness.replace(highlight, "   ")
            val_input_for_sufficiency += (highlight + "   ")
        
        #print("val_input_for_comprehensiveness: ", val_input_for_comprehensiveness)
        #print("val_input_for_sufficiency: ", val_input_for_sufficiency)

        if args.prompt == "two_shot":

            ##=====================================## 
            ##for comprehensiveness and sufficiency##
            prompt_for_comprehensiveness =  "<s>[INST] Please answer whether the movie review is positive or negative and then list the important phrases." +\
                input0_for_comprehensiveness + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "Please answer whether the movie review is positive or negative and then list the important phrases." +\
                          input1_for_comprehensiveness + "[/INST]"  + label_train_1 +\
                            " the important phrases are " + str(nyc_data_train_two[1]['highlight']) + "</s><s>[INST]" +\
                                "please answer whether the movie review is positive or negative and then list the important phrases. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                                    val_input_for_comprehensiveness + "[/INST]"    
            
            prompt_for_sufficiency =  "<s>[INST] Please answer whether the movie review is positive or negative and then list the important phrases." +\
                input0_for_sufficiency + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "Please answer whether the movie review is positive or negative and then list the important phrases." +\
                          input1_for_sufficiency + "[/INST]"  + label_train_1 +\
                            " the important phrases are " + str(nyc_data_train_two[1]['highlight']) + "</s><s>[INST]" +\
                                "please answer whether the movie review is positive or negative and then list the important phrases. Format your response starting with either 'the review is positive' or 'the review is negative. Again, the response should start with either 'the review is positive' or 'the review is negative." +\
                                    val_input_for_sufficiency + "[/INST]"  
            ##=====================================## 
                     
        elif args.prompt == "one_shot":

            ##=====================================## 
            ##for comprehensiveness and sufficiency##
            prompt_for_comprehensiveness =  "<s>[INST] Please answer whether the movie review is positive or negative and then list the important phrases." +\
                input0_for_comprehensiveness + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "please answer whether the movie review is positive or negative and then list the important phrases. Format the response starting with either 'the review is positive' or 'the review is negative. Again, the response should start with either 'the review is positive' or 'the review is negative." +\
                            val_input_for_comprehensiveness + "[/INST]"   
            
            prompt_for_sufficiency =  "<s>[INST] Please answer whether the movie review is positive or negative and then list the important phrases." +\
                input0_for_sufficiency + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "please answer whether the movie review is positive or negative and then list the important phrases. Format the response starting with either 'the review is positive' or 'the review is negative." +\
                            val_input_for_sufficiency + "[/INST]"   
            ##=====================================## 

        elif args.prompt == "zero_shot":

            ##=====================================## 
            ##for comprehensiveness and sufficiency##
            prompt_for_comprehensiveness =  "<s>[INST] Please answer whether the movie review is positive or negative and then report important phrases to explain the reason. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                            val_input_for_comprehensiveness + "[/INST]"    
            
            prompt_for_sufficiency  =  "<s>[INST] Please answer whether the movie review is positive or negative and then report important phrases to explain the reason. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                            val_input_for_sufficiency  + "[/INST]"    
            ##=====================================##      

        ##=====================================## 
        ##for comprehensiveness and sufficiency##
        input_ids_comprehensiveness = tokenizer.encode(prompt_for_comprehensiveness, return_tensors='pt', add_special_tokens=False).to(device)
        input_ids_sufficiency = tokenizer.encode(prompt_for_sufficiency, return_tensors='pt', add_special_tokens=False).to(device)

        model_output_comprehensiveness = model.generate(
            input_ids_comprehensiveness,
            max_length=4096,
            num_return_sequences=1, 
            output_scores=True, 
            return_dict_in_generate=True,
            do_sample=True,
            #eos_token_id=tokenizer.eos_token_id,
        )
        model_output_sufficiency = model.generate(
            input_ids_sufficiency,
            max_length=4096,
            num_return_sequences=1, 
            output_scores=True, 
            return_dict_in_generate=True,
            do_sample=True,
            #eos_token_id=tokenizer.eos_token_id,
        )
        ##=====================================## 

        ##=====================================## 
        ##for comprehensiveness and sufficiency##
        generated_text_comprehensiveness = tokenizer.decode(model_output_comprehensiveness[0][0], skip_special_tokens=True).split("/INST]")[-1]
        generated_text_sufficiency = tokenizer.decode(model_output_sufficiency[0][0], skip_special_tokens=True).split("/INST]")[-1]
        generated_text_comprehensiveness = generated_text_comprehensiveness.strip()
        generated_text_sufficiency = generated_text_sufficiency.strip()
        ##=====================================## 
        
        print("generated_text_comprehensiveness", generated_text_comprehensiveness)
        print("--------------")
        print("generated_text_sufficiency", generated_text_sufficiency)
        print("--------------")

        #print(generated_text)
        nyc_data_val[i]['generated_llama2']=generated_text

        

        ##=====================================## 
        ##for comprehensiveness and sufficiency##
        positive_id = tokenizer.encode('positive', return_tensors='pt', add_special_tokens=False).item()
        negative_id = tokenizer.encode('negative', return_tensors='pt', add_special_tokens=False).item()
    
        full_prompt_prob = 0
        Mask_prompt_prob_comprehensiveness = 0
        Mask_prompt_prob_sufficiency = 0

        print(f"scores: {model_output.scores}") # too many -inf, leads to predict 1.0 for the predicted word
        #score = torch.stack(model_output.scores, dim=1)
        #probs = torch.maximum(score, torch.zeros_like(score)).softmax(-1)
        probs = torch.stack(model_output.scores, dim=1).softmax(-1)
        max_values, max_idxs = torch.max(probs, dim=-1)
        print("max_idxs",max_idxs)
        max_idxs_token = tokenizer.decode(max_idxs[0], skip_special_tokens=True)
        print("max_idxs_token",max_idxs_token)

        for j, word_id in enumerate(max_idxs[0]):
            if  word_id == negative_id:
                idx = negative_id #8178
                full_prompt_prob = probs[0,j,idx] #idx is the index of the positive/negative
                print(f"Negative, Prob: {full_prompt_prob}, Word_id: {word_id}") #negative word_id = 8178
                break
            if  word_id == positive_id:
                idx = positive_id #6374
                full_prompt_prob = probs[0,j,idx] #idx is the index of the positive/negative
                print(f"Positive, Prob: {full_prompt_prob}, Word_id: {word_id}") #positive word_id = 6374
                break

        ##=====================================## 
        score = torch.stack(model_output_comprehensiveness.scores, dim=1)
        probs_comprehensiveness = torch.maximum(score, torch.zeros_like(score)).softmax(-1)
        tokenized_text_comprehensiveness = tokenizer.tokenize(generated_text_comprehensiveness)
        
        score = torch.stack(model_output_sufficiency.scores, dim=1)
        probs_sufficiency = torch.maximum(score, torch.zeros_like(score)).softmax(-1)
        tokenized_text_sufficiency = tokenizer.tokenize(generated_text_sufficiency)
        ##=====================================## 
        #comprehensiveness
        max_values, max_idxs = torch.max(probs_comprehensiveness, dim=-1)
        for j, word_id in enumerate(max_idxs[0]):            
            if word_id == idx:
                Mask_prompt_prob_comprehensiveness = probs_comprehensiveness[0,j,idx] #idx is the index of the positive/negative
                print(f"comprehensiveness, Prob: {Mask_prompt_prob_comprehensiveness}, Word_id: {word_id}") 
                break

        if type(full_prompt_prob) == int or type(full_prompt_prob) == float:
            full_prompt_prob = full_prompt_prob
        else:
            full_prompt_prob = full_prompt_prob.item() 

        if type(Mask_prompt_prob_comprehensiveness) == int or type(Mask_prompt_prob_comprehensiveness) == float:
            Mask_prompt_prob_comprehensiveness = Mask_prompt_prob_comprehensiveness
        else:
            Mask_prompt_prob_comprehensiveness = Mask_prompt_prob_comprehensiveness.item() 

        comprehensiveness = full_prompt_prob - Mask_prompt_prob_comprehensiveness if full_prompt_prob > Mask_prompt_prob_comprehensiveness else 0
        #print("full_prompt_prob",full_prompt_prob)
        #print("Mask_prompt_prob_comprehensiveness",Mask_prompt_prob_comprehensiveness)


        #sufficiency
        max_values, max_idxs = torch.max(probs_sufficiency, dim=-1)
        for j, word_id in enumerate(max_idxs[0]):            
            if word_id == idx:
                Mask_prompt_prob_sufficiency = probs_sufficiency[0,j,idx] #idx is the index of the positive/negative
                print(f"sufficiency, Prob: {Mask_prompt_prob_sufficiency}, Word_id: {word_id}") 
                break


        if type(Mask_prompt_prob_sufficiency) == int or type(Mask_prompt_prob_sufficiency) == float:
            Mask_prompt_prob_sufficiency = Mask_prompt_prob_sufficiency
        else:
            Mask_prompt_prob_sufficiency = Mask_prompt_prob_sufficiency 

        sufficiency = full_prompt_prob - Mask_prompt_prob_sufficiency if full_prompt_prob > Mask_prompt_prob_sufficiency else 0
        #print("full_prompt_prob",full_prompt_prob)
        #print("Mask_prompt_prob_comprehensiveness",Mask_prompt_prob_sufficiency)


        ##=====================================## 
        comprehensiveness_list.append(comprehensiveness)
        sufficiency_list.append(sufficiency)
        nyc_data_val[i]['comprehensiveness'] = comprehensiveness
        nyc_data_val[i]['sufficiency'] = sufficiency


    from numpy import mean, std
    #average_iou_f1 = mean(iou_f1_scores) 
    #average_token_f1 = mean(token_f1_scores) 
    #average_comprehensiveness = mean(comprehensiveness_list) 
    #average_sufficiency = mean(sufficiency_list) 
    print("one_shot")
    print(f"Average IOU F1 Score: {mean(iou_f1_scores) }, standard deviation: {std(iou_f1_scores)}")
    print(f"Average Token F1 Score: {mean(token_f1_scores)}, standard deviation: {std(token_f1_scores)}")
    print(f"Average comprehensiveness: {mean(comprehensiveness_list) }, standard deviation: {std(comprehensiveness_list)}")
    print(f"Average sufficiency: {mean(sufficiency_list)}, standard deviation: {std(sufficiency_list)}")

    filename = 'out/val_' + args.prompt + '.jsonl'
    with jsonlines.open(filename, mode='w') as writer:
        for item in nyc_data_val:
            writer.write(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default = 1418772, type=int, help='Random seed set to your uNID') # <======================> 
    parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
    parser.add_argument('--cache_dir', type=str, help='Directory where cache will be saved')
    parser.add_argument('--task_name', default="movie_rationales",  type=str, help='Name of the task that will be used by huggingface load dataset')    
    #parser.add_argument('--subtask', default="explanation", type=str, help="The contest has three subtasks: matching, ranking, explanation")
    parser.add_argument('--llama2_checkpoint', default="meta-llama/Llama-2-7b-chat-hf", type=str, help="The hf name of a llama2 checkpoint")
    parser.add_argument('--val_size', default=200, type=int, help="The sample size of validation dataset.")
    parser.add_argument('--prompt', default="one_shot", type=str, help="Control the type of prompt.")
    args = parser.parse_args()
    if args.prompt not in ["zero_shot", "one_shot", "two_shot"]:
        raise ValueError("Arg \"-prompt\" should be \"zero_shot\", \"one_shot\", or \"two_shot\"")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    movie_rationales_llama2(args)