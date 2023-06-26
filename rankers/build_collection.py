import os

DATA_PATH = 'data/qasper/collections/all/'
for file in os.listdir(DATA_PATH):
    os.system(f"python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input {DATA_PATH}/{file} \
  --index {DATA_PATH}/{file} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw")