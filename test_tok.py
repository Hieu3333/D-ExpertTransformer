import json
with open('data/cleaned_DeepEyeNet_test.json','r') as f:
    ann = json.load(f)
print(len(ann))