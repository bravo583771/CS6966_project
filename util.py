__all__  = ['iou_f1_score', 'token_f1_score', 'movie_rationales_data', 'label4prompt'] 

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
    
def token_f1_score(model_highlights, human_highlights):
    # Convert highlights to sets of tokens
    model_tokens = set(model_highlights.split())
    human_tokens = set(human_highlights.split())

    # Calculate token precision and recall
    common_tokens = model_tokens.intersection(human_tokens)
    token_precision = len(common_tokens) / len(model_tokens) if len(model_tokens) > 0 else 0
    token_recall = len(common_tokens) / len(human_tokens) if len(human_tokens) > 0 else 0

    # Calculate Token F1 score
    if token_precision + token_recall == 0:
        return 0.0  # Handle the case of zero precision and recall
    else:
        token_f1 = 2 * (token_precision * token_recall) / (token_precision + token_recall)
        return token_f1

def movie_rationales_data(task_name = 'movie_rationales'):
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
    #return "yes" if label==1 else "no" 
    #return "Yes, the movie review is positive." if label==1 else "No, the movie review is negative." 
    #return "positive" if label==1 else "negative" 
    return "The movie review is " + ("positive." if label==1 else "negative.")