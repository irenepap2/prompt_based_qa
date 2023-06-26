import json
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

test = json.load(open("./data/qasper/qasper-test-v0.3.json",'r'))

model_name = "castorini/monot5-base-msmarco-10k"
# model = T5ForConditionalGeneration.from_pretrained('castorini/monot5-base-msmarco-10k')
model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
top_k = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

reranker = MonoT5(model=model)

monoT5_results = {}
for k, v in tqdm(test.items()):
    texts = []
    for content in v['full_text']:
        section_name = content['section_name']
        section_paragraphs = content["paragraphs"]
        for i, paragraph in enumerate(section_paragraphs):
            section_name = "" if section_name is None else section_name
            texts.append(Text(paragraph, {'docid': k + '_' + section_name + '_' + str(i), "contents" : paragraph}, 0))
    
    for qa in tqdm(v['qas']):
        query = Query(qa['question'])
        reranked = reranker.rerank(query, texts)
        reranked.sort(key=lambda x: x.score, reverse=True)
        monoT5_results[qa['question_id']] = [x.metadata for x in reranked[:top_k]]

monoT5_contents = {}
for k, v in monoT5_results.items():
    monoT5_contents[k] = [x["contents"] for x in v]

# Save monoT5 results
filename = model_name.split("/")[-1]
with open(f'./retrieval_passages/{filename}_results.json', 'w') as f:
    json.dump(monoT5_results, f, indent=4)

with open(f'./retrieval_passages/{filename}_contents.json', 'w') as f:
    json.dump(monoT5_contents, f, indent=4)