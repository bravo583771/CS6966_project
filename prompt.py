import argparse
import numpy as np
import torch
import jsonlines 
import random 
import os
#from sklearn.metrics import f1_score

from util import *

def movie_rationales_llama2(args): 
    print("Loading data")
    nyc_data = movie_rationales_data(args.task_name)
    nyc_data_train_two = random.sample(nyc_data['train'], 2)
    nyc_data_five_val = random.sample(nyc_data['val'], args.val_size)
    
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

    print("Loading model")
    import transformers
    from transformers import AutoTokenizer
    access_token = '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'
    tokenizer = AutoTokenizer.from_pretrained(args.llama2_checkpoint, cache_dir = args.cache_dir, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(args.llama2_checkpoint, return_dict_in_generate=True, token=access_token)

    iou_f1_scores = []
    token_f1_scores = []

    ##For Faithfulness##
    #calculate Sufficiency & Comprehensiveness
    input0_for_faithfulness = nyc_data_train_two[0]['input']
    for highlight in nyc_data_train_two[0]['highlight']:
        input0_for_faithfulness.replace(highlight, " ")

    input1_for_faithfulness = nyc_data_train_two[1]['input']
    for highlight in nyc_data_train_two[1]['highlight']:
        input1_for_faithfulness.replace(highlight, " ")


    for i, val_inst in enumerate(nyc_data_five_val):         
        # ======================> ADD YOUR CODE TO DEFINE A PROMPT WITH TWO TRAIN EXAMPLES/DEMONSTRATIONS/SHOTS <======================
        label_train_0 = label4prompt(nyc_data_train_two[0]['target']) # 0 = negative, 1 = positive
        label_train_1 = label4prompt(nyc_data_train_two[1]['target']) # 0 = negative, 1 = positive
        
        """
        ##For Faithfulness##
        val_input_for_faithfulness = val_inst['input']
        for highlight in val_input_for_faithfulness['highlight']:
            val_input_for_faithfulness.replace(highlight, "")
        """

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
                                "According to the above two examples, please answer whether the movie review is positive or negative and then list the important phrases. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                                    val_inst['input'] + "[/INST]"              
            
            """
            prompt_for faithfulness =  "<s>[INST] Please answer whether the movie review is positive or negative and then list the important phrases." +\
                input0_for_faithfulness + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "Please answer whether the movie review is positive or negative and then list the important phrases." +\
                          input1_for_faithfulness + "[/INST]"  + label_train_1 +\
                            " the important phrases are " + str(nyc_data_train_two[1]['highlight']) + "</s><s>[INST]" +\
                                "According to the above two examples, please answer whether the movie review is positive or negative and then list the important phrases. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                                    val_input_for_faithfulness + "[/INST]"    
            """
                     
        elif args.prompt == "one_shot":

            
            # 1: 0.194 for 5 val instances
            #Average IOU F1 Score: 0.23042911440679834 for 200 instances
            #Average Token F1 Score: 0.2304291144067983 for 200 instances
            prompt =  "<s>[INST] Please answer whether the movie review is positive or negative and then report important phrases to explain the reason." +\
                nyc_data_train_two[0]['input'] + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "According to the above example, please answer whether the movie review is positive or negative and then report important phrases to explain the reason. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                            val_inst['input'] + "[/INST]"                
            
            """
            prompt_for faithfulness =  "<s>[INST] Please answer whether the movie review is positive or negative and then list the important phrases." +\
                input0_for_faithfulness + "[/INST]" + label_train_0 +\
                    " the important phrases are " + str(nyc_data_train_two[0]['highlight']) + "</s><s>[INST]" +\
                        "According to the above two examples, please answer whether the movie review is positive or negative and then list the important phrases. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                            val_input_for_faithfulness + "[/INST]"    
            """

        
            '''
            #2: 0.21088923922994768 for 5 val instances
            #Even though the iou score is higher, the answer is incorrect 
            #Yes, the movie is nagative -----should be ----->  No, the movie is nagative
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

            prompt =  "<s>[INST] Please answer whether the movie review is positive or negative and then report important phrases to explain the reason. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                            val_inst['input'] + "[/INST]"  

            """
            prompt_for faithfulness =  "<s>[INST] Please answer whether the movie review is positive or negative and then report important phrases to explain the reason. Format your response starting with either 'the review is positive' or 'the review is negative." +\
                            val_input_for_faithfulness + "[/INST]"    
            """              
            

        max_token_limit = 4096
        """
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

        #input_ids = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).input_ids
        with torch.no_grad():
            generated_outputs = model.generate(input_ids,do_sample=True, num_return_sequences=1, output_scores=True, return_dict_in_generate=True,) 
        print(generated_outputs)
        
        prob = torch.stack(generated_outputs.scores, dim=1).softmax(-1)
        print(prob)
        print(prob.shape)
        max_values, max_idxs = torch.max(prob, dim=-1) 
        """
        #need the ids for negative and positive to compute faithfulness
        """
        
        print(max_values.shape)
        print(max_idxs.shape)
        # -> shape [1, tokens_size, vocab_size]
        generated_text = tokenizer.decode(generated_outputs[0][0], skip_special_tokens=True).split("/INST]")[-1]
        print(generated_text)
        tokenized_text = tokenizer.tokenize(generated_text)
        for token, prob in zip(tokenized_text, max_values[0]):
            print(f"Token: {token}, Prob: {prob.item()}")

        
        """
        ##For Faithfulness##
        full_prompt_prob = 1
        Mask_prompt_prob = 1
        for i ,token in enumerate(tokenized_text):
            if nyc_data_five_val[i]['target'] ==0:
                goal = "negative"
                idx = 
            elif nyc_data_five_val[i]['target'] ==1:
                goal = "positive"
                idx = 
            
            if goal in token:
                full_prompt_prob = prob[0,i,idx] #idx is the index of the positive/negative
                break
                
        for i ,token in enumerate(mask_tokenized_text):
            if nyc_data_five_val[i]['target'] ==0:
                goal = "negative"
                idx = 
            elif nyc_data_five_val[i]['target'] ==1:
                goal = "positive"
                idx = 
            
            if goal in token:
                Mask_prompt_prob = mask_prob[0,i,idx] #idx is the index of the positive/negative
                break
        
        faithfulness = full_prompt_prob - Mask_prompt_prob
        """
        


        ##Plausibility##
        #calculate Intersection-over-Union (IOU) F1 anf Token F1
        label_val = label4prompt(nyc_data_five_val[i]['target']) # 0 = negative, 1 = positive
        #target_explanation = label_val + " the important phrases are " + str(nyc_data_five_val[i]['highlight'])
        target_explanation = label_val + " the important phrases are " + ' '.join(nyc_data_five_val[i]['highlight'])
        iou_f1 = iou_f1_score(generated_text, target_explanation)
        nyc_data_five_val[i]['iou_f1_score'] = iou_f1
        token_f1 = token_f1_score(generated_text, target_explanation)
        nyc_data_five_val[i]['token_f1_score'] = token_f1

        iou_f1_scores.append(iou_f1)
        token_f1_scores.append(token_f1)
        #print("Instance {} done.".format(i))

    average_iou_f1 = sum(iou_f1_scores) / len(iou_f1_scores)
    average_token_f1 = sum(token_f1_scores) / len(token_f1_scores)
    print("Average IOU F1 Score:", average_iou_f1)
    print("Average Token F1 Score:", average_token_f1)

    #filename = 'out/val_' + args.prompt + '.jsonl'
    #with jsonlines.open(filename, mode='w') as writer:
    #    for item in nyc_data_five_val:
    #        writer.write(item)

    #filename = 'out/train.jsonl'
    #with jsonlines.open(filename, mode='w') as writer:
    #    for item in nyc_data_train_two:
    #        writer.write(item)


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