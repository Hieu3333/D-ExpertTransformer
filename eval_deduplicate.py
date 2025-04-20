import json
from modules.metrics import compute_scores
with open('logs/resnet-deepeyenet-diff/test_result_epoch_50.json','r') as f:
    data = json.load(f)

gt_set = set()
gt={}
res={}
for entry in data:
    if entry['ground_truth'] not in gt_set:
        gt_set.add(entry['ground_truth'])
        id = entry['image_id']
        ground_truth = entry['ground_truth']
        generated_caption = entry['generated_caption']
        gt[id] = [ground_truth]
        res[id] = [generated_caption]

scores = compute_scores(gt,res)
print(f"BLEU_1: {scores['BLEU_1']}")
print(f"BLEU_2: {scores['BLEU_2']}")
print(f"BLEU_3: {scores['BLEU_3']}")
print(f"BLEU_4: {scores['BLEU_4']}")
print(f"METEOR: {scores['METEOR']}")
print(f"CIDER: {scores['Cider']}")
print(f"ROUGE_L: {scores['ROUGE_L']}")