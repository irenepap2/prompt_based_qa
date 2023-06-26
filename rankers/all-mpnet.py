import json
import torch

from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

test = json.load(open("./data/qasper/qasper-test-v0.3.json",'r'))

model_name = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(model_name)
top_k = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

allmpent_results = {}
for k, v in tqdm(test.items()):
    sentences = []
    for content in v['full_text']:
        section_name = content['section_name']
        section_paragraphs = content["paragraphs"]
        for i, paragraph in enumerate(section_paragraphs):
            section_name = "" if section_name is None else section_name
            sentences.append(paragraph)
    passage_embeddings = model.encode(sentences)
    
    # queries = []
    for qa in tqdm(v['qas']):
        query = qa['question']
        # queries.append(query)
        query_embedding = model.encode(query)
        similarity_scores = util.dot_score(query_embedding, passage_embeddings)
        ranked_indices = similarity_scores.argsort(descending=True)
        allmpent_results[qa['question_id']] = [sentences[i] for i in ranked_indices[0][:top_k]]

# Save monoT5 results
filename = model_name.split("/")[-1]
with open(f'./retrieval_passages/{filename}_contents.json', 'w') as f:
    json.dump(allmpent_results, f, indent=4)