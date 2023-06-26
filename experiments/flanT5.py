from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import os
from qasper_evaluator import get_answers_and_evidence
from tqdm import tqdm
import torch
from pipeline import utils

# check if GPU is available
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

model_name = "google/flan-t5-xxl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
filename = model_name.split("/")[-1]
prompt_filename = "qasper_fewshot_prompt"

BATCH_SIZE = 4
DATA_PATH = 'data/qasper/'
RETRIEVAL_PATH = 'retrieval_passages/'
RETRIEVAL_METHOD = 'monot5-base-msmarco-10k'
RESULT_PATH = 'results/'

# make dir if not exist
os.makedirs(os.path.join(RESULT_PATH, RETRIEVAL_METHOD, filename), exist_ok=True)
RESULTS_FILE = os.path.join(RESULT_PATH, RETRIEVAL_METHOD, filename, f'{filename}_{prompt_filename}.json')
PRED_FILE = os.path.join(RESULT_PATH, RETRIEVAL_METHOD, filename, f'{filename}_{prompt_filename}.jsonl')

def run_model(input_string, **generator_args):
    inputs = tokenizer(
        input_string, 
        return_tensors="pt", 
        # max_length=512, 
        truncation=True, 
        padding=True
    )
    res = model.generate(
        inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"], 
        max_new_tokens=20,
        # min_new_tokens= 1,
        num_beams=5,
        temperature=0,
        no_repeat_ngram_size=1,
        **generator_args
    )
    return tokenizer.batch_decode(res, skip_special_tokens=True)

def get_predictions(answers_and_evidence, gen_answers):
    predictions = []
    for q_id, references in answers_and_evidence.items():
        predicted_answer = gen_answers[q_id]
        predicted_answer = "Yes" if predicted_answer.startswith("Yes") else "No" if predicted_answer.startswith("No") else predicted_answer
        golden_answers = [i['answer'] for i in references]
        if RETRIEVAL_METHOD == "gold":
            predicted_evidence = retrieval_psgs[q_id]
        else:
            predicted_evidence = [i['evidence'] for i in references] # gold evidence
        predictions.append({'question_id': q_id, 'predicted_answer': predicted_answer, 'golden_answers': golden_answers, 'predicted_evidence': predicted_evidence})
    return predictions

with open(os.path.join(DATA_PATH, 'qasper-test-v0.3.json'), 'r') as f:
    test = json.load(f)

test_questions = {}
for k, v in test.items():
    for qa in v['qas']:
        q_id = qa['question_id']
        test_questions[q_id] = qa['question']

test_answers_and_evidence = get_answers_and_evidence(test, text_evidence_only=True)

if RETRIEVAL_METHOD != "gold":
    with open(os.path.join(RETRIEVAL_PATH, f'{RETRIEVAL_METHOD}_contents.json'), 'r') as f:
        retrieval_psgs = json.load(f)

# Load generated answers if file exists
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r') as f:
        gen_answers = json.load(f)
else:
    gen_answers = {}

with torch.no_grad():
    N = len(test_questions)
    for i in tqdm(range(0, N, BATCH_SIZE)):
        print(f"Batch #{(i // BATCH_SIZE)+1} of {(N // BATCH_SIZE)+1}")

        q_ids = list(test_questions)[i:i+BATCH_SIZE]

        # check if answers for these questions are already generated
        ids = [q_id for q_id in q_ids if q_id not in gen_answers.keys()]
        if len(ids) == 0:
            print(f"Answers for {q_ids} already generated")
            continue

        questions = [test_questions[q_id] for q_id in ids]
        if RETRIEVAL_METHOD == "gold":
            contexts = [" | ".join([text for text in test_answers_and_evidence[q_id][0]['evidence'] if "FLOAT SELECTED" not in text]) for q_id in ids]
        else:
            contexts = [" | ".join([text for text in retrieval_psgs[q_id][:3] if "FLOAT SELECTED" not in text]) for q_id in ids]
        # input_string = [f"Read this and answer the question. If the question is unanswerable, say \"Unanswerable\". Answer only \"Yes\" or \"No\" in boolean questions.\n\n{context.lower()}\n\n{question.lower()}" for question, context in zip(questions, contexts)]
        # input_string = [f"Answer this question given the context. If the question is unanswerable, say \"Unanswerable\". Answer only \"Yes\" or \"No\" in boolean questions. {question.lower()} \\n {context.lower()}" for question, context in zip(questions, contexts)]
        # input_string = [f"{prompt} {question.lower()} \\n {context.lower()}" for question, context in zip(questions, contexts)]
        input_string = [
            utils.construct_prompt(filename=f"{prompt_filename}.txt",
                prompt_params_dict={
                    "question": question,
                    "context": context,
                }
            ) for question, context in zip(questions, contexts)]
        # print(input_string[0])
        # print(input_string[1])
        outputs = run_model(input_string)

        for q_id, ans in zip(ids, outputs):
            gen_answers[q_id] = ans
            
        # save the generated answers
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(gen_answers, f, indent=4)

# replace no answer with Unanswerable
# for k, v in gen_answers.items():
#     if v == "no answer>":
#         gen_answers[k] = "Unanswerable"

predictions = get_predictions(test_answers_and_evidence, gen_answers)

# save the predictions in JSON lines format
with open(PRED_FILE, 'w') as f:
    for pred in predictions:
        f.write(json.dumps(pred) + '\n')
