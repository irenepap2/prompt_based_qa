import json
import os
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

DATA_PATH = '../data/qasper/'
INDEX_PATH = '../data/qasper/collections/all/'

#Load the data
TRAIN_FILE = os.path.join(DATA_PATH, 'qasper-train-v0.3.json')
DEV_FILE = os.path.join(DATA_PATH, 'qasper-dev-v0.3.json')
TEST_FILE = os.path.join(DATA_PATH, 'qasper-test-v0.3.json')

with open(TRAIN_FILE, 'r') as f:
    train = json.load(f)

with open(DEV_FILE, 'r') as f:
    dev = json.load(f)

with open(TEST_FILE, 'r') as f:
    test = json.load(f)

# join all dicts into one
all = train | dev | test

# Gather all questions
all_questions = {}
for k, v in all.items():
    for qa in v['qas']:
        q_id = qa['question_id']
        all_questions[q_id] = {"question" : qa['question'], "paper_id": k}

bm25_results = {}

for k, v in tqdm(all_questions.items()):
    question = v["question"]
    paper_id = v["paper_id"]
    if os.path.exists(f"{INDEX_PATH}/{paper_id}"):
        searcher = LuceneSearcher(f"{INDEX_PATH}/{paper_id}")
        hits = searcher.search(question, k=5)

    bm25_results[k] = []
    for i in range(len(hits)):
        bm25_results[k].append(hits[i].docid)

os.makedirs('../retrieval_passages', exist_ok=True)
with open('../retrieval_passages/bm25_results.json', 'w') as f:
    json.dump(bm25_results, f, indent=4)

# Turn doc_ids into content
bm25_contents = {}
for q_id, results in tqdm(bm25_results.items()):
    bm25_contents[q_id] = []
    for result in results:
        paper_id = result.split('_')[0]
        with open(f'../data/qasper/collections/all/{paper_id}/{paper_id}.jsonl', 'r') as f:
            for line in f:
                passage = json.loads(line)
                if passage["id"] == result:
                    bm25_contents[q_id].append(passage["contents"])

#save bm25 contents
with open('../retrieval_passages/bm25_contents.json', 'w') as f:
    json.dump(bm25_contents, f, indent=4)