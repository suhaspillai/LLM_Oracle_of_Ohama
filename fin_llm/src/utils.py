
def parse_answer(answer):
    import re
    
    match_res = re.match(r"^\s*\[Positive Developments\]:\s*(.*)\s*\[Potential Concerns\]:\s*(.*)\s*\[Prediction & Analysis\]:\s*(.*)\s*$", answer, flags=re.DOTALL)
    if not match_res:
        return None
    
    pros, cons, pna = match_res.group(1), match_res.group(2), match_res.group(3)
        
    match_res = re.match(r'^Prediction:\s*(.*)\s*Analysis:\s*(.*)\s*$', pna, flags=re.DOTALL)
    if not match_res:
        return None
        
    pred, anal = match_res.group(1), match_res.group(2)
        
    if re.search(r'up|increase', pred.lower()):
        pred_bin = 1
    elif re.search(r'down|decrease|decline', pred.lower()):
        pred_bin = -1
    else:
        pred_bin = 0
            
    match_res = re.search(r'(\d)-(\d)%', pred)
    if not match_res:
        match_res = re.search(r'(?:more than )?(\d)+?%', pred)    
        
    pred_margin = pred_bin * (int(match_res.group(1)) + 0.5) if match_res else 0.
        
    return {
        "positive developments": pros,
        "potential concerns": cons,
        "prediction": pred_margin,
        "prediction_binary": pred_bin,
        "analysis": anal
    }
    

def calc_rouge_score(references, answers):
    from rouge_score import rouge_scorer
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    scores_per_pair = [scorer.score(ref, ans) for ref, ans in zip(references, answers)]
    
    rouge1 = sum(score['rouge1'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rouge2 = sum(score['rouge2'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rougeL = sum(score['rougeL'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    
    return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}

    
def calc_metrics(answers, gts):
    from collections import defaultdict
    
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    
    for answer, gt in zip(answers, gts):
        answer_dict = parse_answer(answer)
        gt_dict = parse_answer(gt)
        
        if answer_dict and gt_dict:
            for k in answer_dict.keys():
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])
    
    if not answers_dict['prediction']:
        return {}
    
    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])
    
    pros_rouge_scores = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge_scores = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge_scores = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])
                              
    print(f"\nBinary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
    print(f"\nRouge Score of Positive Developments: {pros_rouge_scores}")
    print(f"\nRouge Score of Potential Concerns: {cons_rouge_scores}")
    print(f"\nRouge Score of Summary Analysis: {anal_rouge_scores}")
                              
    return {
        "valid_count": len(answers_dict['prediction']),
        "bin_acc": bin_acc,
        "mse": mse,
        "pros_rouge_scores": pros_rouge_scores,
        "cons_rouge_scores": cons_rouge_scores,
        "anal_rouge_scores": anal_rouge_scores
    }