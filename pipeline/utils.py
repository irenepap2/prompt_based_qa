from datetime import datetime, timedelta
from jinja2 import Template
from dotenv import load_dotenv
from pipeline.constants import PROMPT_PATH, ANSWER_PATH

import os
import openai
import json
import backoff
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_current_date():
    '''
    Get current date and time in this format %Y-%m-%dT%H:%M:%S
    Returns:
        dt_string (str): current date and time in this format %Y-%m-%dT%H:%M:%S
    '''
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H:%M:%S")


def get_date_five_years_ago():
    '''
    Get date and time in this format %Y-%m-%dT%H:%M:%S 5 years ago (for few shot prompt)
    Returns:
        dt_string (str): current date and time in this format %Y-%m-%dT%H:%M:%S
    '''
    # get current date and time in this format %Y-%m-%dT%H:%M:%S
    now = datetime.now()
    five_years_ago = now - timedelta(days=1825)
    return five_years_ago.strftime("%Y-%m-%dT%H:%M:%S")


def construct_prompt(filename, prompt_params_dict):
    '''
    Construct prompt from template
    Args:
        filename (str): filename of prompt template
        prompt_params_dict (dict): dictionary of prompt parameters
    Returns:
        prompt (str): prompt to send to OpenAI
    '''
    with open(os.path.join(PROMPT_PATH, filename), "r") as f:
        prompt = f.read()
    return Template(prompt).render(**prompt_params_dict)


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_model_response(prompt, model):
    '''
    Get response from OpenAI
    Args:
        prompt (str): prompt to send to OpenAI
        model (str): model to use
    Returns:
        response (str): response from OpenAI
    '''
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["context:", "Example", "question:"],
    )

    return response

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_chat_model_response(context, question, prompt_filename, additional_response=""):

    # prompt = construct_prompt(f"{prompt_filename}.txt", {"context": context, "question": question})
    # print(prompt)
    # context for CoT prompt
    context_for_prompt = [f"[Document {i+1}]: " + con for i, con in enumerate(context)]
    context_for_prompt = "\n".join(context_for_prompt)
    print(context_for_prompt)
    
    # construct prompt
    system_instruction = "You are a highly intelligent question answering bot. Answer the following question given the context. If I ask you a question that is nonsense, trickery, or has no clear answer, you should respond strictly with \"Unanswerable\" and nothing else."
    # instruction = "If I ask you a question that is nonsense, trickery, or has no clear answer, you should respond strictly with \"Unanswerable\" and nothing else."
    # context_and_question = "context: " + context + "\nquestion: " + question + " " + instruction
    # print(context_and_question)

    # all in user CoT prompt
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when not enough information is provided in the documents. Pay attention to answer only \"Yes\" or \"No\" in boolean questions."},
    #         {"role": "user", "content": prompt + additional_response}
    #         ],
    #     temperature=0,
    #     max_tokens=256,
    # )

    # user-assistant CoT prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Answer \"Unanswerable\" when not enough information is provided in the documents. Pay attention to answer only \"Yes\" or \"No\" in boolean questions."},
            {"role": "user", "content": '''Example 1:
[Document 1]: The seed lexicon consists of positive and negative predicates. If the predicate of an extracted event is in the seed lexicon and does not involve complex phenomena like negation, we assign the corresponding polarity score ($+1$ for positive events and $-1$ for negative events) to the event. We expect the model to automatically learn complex phenomena through label propagation. Based on the availability of scores and the types of discourse relations, we classify the extracted event pairs into the following three types.
[Document 2]: Although event pairs linked by discourse analysis are shown to be useful, they nevertheless contain noises. Adding linguistically-motivated filtering rules would help improve the performance.
Question: What is the seed lexicon?
'''},
            {"role": "assistant", "content": '''Explanation: The seed lexicon is defined in [Document 1] as a vocabulary of positive and negative predicates. This lexicon is used to determine the polarity score of an event.
Answer: a vocabulary of positive and negative predicates that helps determine the polarity score of an event
'''},
            {"role": "user", "content": '''Example 2:
[Document 1]: Awe/Sublime (found it overwhelming/sense of greatness): Awe/Sublime implies being overwhelmed by the line/stanza, i.e., if one gets the impression of facing something sublime or if the line/stanza inspires one with awe (or that the expression itself is sublime). Such emotions are often associated with subjects like god, death, life, truth, etc. The term Sublime originated with kant2000critique as one of the first aesthetic emotion terms. Awe is a more common English term.
Question: Does the paper report macro F1?'''},
            {"role": "assistant", "content": '''Explanation: [Document 1] reports that the term "Sublime" originated with kant2000critique. This information is further supported by [Document 2], which reports that kant2000critique is a paper that discusses the macro F1 score.
Answer: Yes'''},
            {"role": "user", "content": '''Example 3:
[Document 1]: In this section, we devote to experimentally evaluating our proposed task and approach. The best results in tables are in bold.
Question: Are there privacy concerns with clinical data?'''},
            {"role": "assistant", "content": '''Explanation: The question cannot be answered with the information provided in the document.
Answer: Unanswerable'''},
            {"role": "user", "content": '''Example 4:
[Document 1]: Experimental results for the Twitter Sentiment Classification task on Kaggle's Sentiment140 Corpus dataset, displayed in Table TABREF37, show that our model has better F1-micros scores, outperforming the baseline models by 6$%$ to 8$%$. We evaluate our model and baseline models on three versions of the dataset. The first one (Inc) only considers the original data, containing naturally incorrect tweets, and achieves accuracy of 80$%$ against BERT's 72$%$. The second version (Corr) considers the corrected tweets, and shows higher accuracy given that it is less noisy. In that version, Stacked DeBERT achieves 82$%$ accuracy against BERT's 76$%$, an improvement of 6$%$. In the last case (Inc+Corr), we consider both incorrect and correct tweets as input to the models in hopes of improving performance. However, the accuracy was similar to the first aforementioned version, 80$%$ for our model and 74$%$ for the second highest performing model. Since the first and last corpus gave similar performances with our model, we conclude that the Twitter dataset does not require complete sentences to be given as training input, in addition to the original naturally incorrect tweets, in order to better model the noisy sentences.
[Document 2]: The reconstructed hidden sentence embedding $h_{rec}$ is compared with the complete hidden sentence embedding $h_{comp}$ through a mean square error loss function, as shown in Eq. (DISPLAY_FORM7)
[Document 3]: Experimental results for the Intent Classification task on the Chatbot NLU Corpus with STT error can be seen in Table TABREF40. When presented with data containing STT error, our model outperforms all baseline models in both combinations of TTS-STT: gtts-witai outperforms the second placing baseline model by 0.94% with F1-score of 97.17%, and macsay-witai outperforms the next highest achieving model by 1.89% with F1-score of 96.23%.
Question: By how much do they outperform other models in the sentiment in intent classification tasks?'''},
            {"role": "assistant", "content": '''Explanation: According to [Document 1], the model outperforms the baseline models by 6% to 8% in the sentiment classification task. This information is further supported by [Document 3], which states that the model outperforms the baseline models by 0.94% on average in the intent classification task.
Answer: In the sentiment classification task by 6% to 8% and in the intent classification task by 0.94% on average
'''},
            {"role": "user", "content": f'''Example 5:
{context_for_prompt}
Question: {question}''' + additional_response},
            ],
        temperature=0,
        max_tokens=256,
    )

    # all in user fewshot prompt
    # response = openai.ChatCompletion.create(
    #     model = "gpt-3.5-turbo",
    #     messages = [
    #         {"role": "system", "content": system_instruction},
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature = 0,
    #     max_tokens = 256,
    # )
    # print(response)

    # user-assistant fewshot prompt
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #             {"role": "system", "content": "Answer the following question given the context. " + system_instruction},
    #             {"role": "user", "content": "context: The DDI corpus contains thousands of XML files, each of which are constructed by several records. For a sentence containing INLINEFORM0 drugs, there are INLINEFORM1 drug pairs. We replace the interested two drugs with \u201cdrug1\u201d and \u201cdrug2\u201d while the other drugs are replaced by \u201cdurg0\u201d, as in BIBREF9 did. This step is called drug blinding. For example, the sentence in figure FIGREF5 generates 3 instances after drug blinding: \u201cdrug1: an increased risk of hepatitis has been reported to result from combined use of drug2 and drug0\u201d, \u201cdrug1: an increased risk of hepatitis has been reported to result from combined use of drug0 and drug2\u201d, \u201cdrug0: an increased risk of hepatitis has been reported to result from combined use of drug1 and drug2\u201d. The drug blinded sentences are the instances that are fed to our model. \n question: How big is the evaluated dataset? Answer only \"Yes\" or \"No\" in boolean questions, and answer \"Unanswerable\" if you don't know."},
    #             {"role": "assistant", "content": "contains thousands of XML files, each of which are constructed by several records"},
    #             {"role": "user", "content": "Our full dataset consists of all subreddits on Reddit from January 2013 to December 2014, for which there are at least 500 words in the vocabulary used to estimate our measures, in at least 4 months of the subreddit's history. We compute our measures over the comments written by users in a community in time windows of months, for each sufficiently active month, and manually remove communities where the bulk of the contributions are in a foreign language. This results in 283 communities, for a total of 4,872 community-months. \n question: Do they report results only on English data? Answer only \"Yes\" or \"No\" in boolean questions, and answer \"Unanswerable\" if you don't know."},
    #             {"role": "assistant", "content": "No"},
    #             {"role": "user", "content": "context: \nquestion: How does using NMT `ensure generated reviews stay on topic? Answer only \"Yes\" or \"No\" in boolean questions, and answer \"Unanswerable\" if you don't know."},
    #             {"role": "assistant", "content": "Unanswerable"},
    #             {"role": "user", "content": "Experimental results for the Twitter Sentiment Classification task on Kaggle's Sentiment140 Corpus dataset, displayed in Table TABREF37, show that our model has better F1-micros scores, outperforming the baseline models by 6$\\%$ to 8$\\%$. We evaluate our model and baseline models on three versions of the dataset. The first one (Inc) only considers the original data, containing naturally incorrect tweets, and achieves accuracy of 80$\\%$ against BERT's 72$\\%$. The second version (Corr) considers the corrected tweets, and shows higher accuracy given that it is less noisy. In that version, Stacked DeBERT achieves 82$\\%$ accuracy against BERT's 76$\\%$, an improvement of 6$\\%$. In the last case (Inc+Corr), we consider both incorrect and correct tweets as input to the models in hopes of improving performance. However, the accuracy was similar to the first aforementioned version, 80$\\%$ for our model and 74$\\%$ for the second highest performing model. Since the first and last corpus gave similar performances with our model, we conclude that the Twitter dataset does not require complete sentences to be given as training input, in addition to the original naturally incorrect tweets, in order to better model the noisy sentences. | Experimental results for the Intent Classification task on the Chatbot NLU Corpus with STT error can be seen in Table TABREF40. When presented with data containing STT error, our model outperforms all baseline models in both combinations of TTS-STT: gtts-witai outperforms the second placing baseline model by 0.94% with F1-score of 97.17%, and macsay-witai outperforms the next highest achieving model by 1.89% with F1-score of 96.23%. \n question: By how much do they outperform other models in the sentiment in intent classification tasks? Answer only \"Yes\" or \"No\" in boolean questions, and answer \"Unanswerable\" if you don't know."},
    #             {"role": "assistant", "content": "In the sentiment classification task by 6% to 8% and in the intent classification task by 0.94% on average"},
    #             {"role": "user", "content": context_and_question},
    #         ],
    #     temperature=0,
    #     max_tokens=256,
    # )
    
    return response["choices"][0]["message"]["content"]


def parse_search_filter_extract_response(response):
    '''
    Parse the response from OpenAI to get the search query, filter query, and extract query
    Args: 
        response (str): response from OpenAI
    Returns:
        search_q (str): search query
        filter_q_dict (dict): dictionary of filter queries
        extract_q (str): extract query
    '''
    queries = response.split("\n")
    search_q = queries[0].removeprefix(" ")
    filter_q_dict = eval(queries[1].removeprefix("Filter query: "))

    # sort_order is a special case
    if filter_q_dict.get("sort_order"):
        sort_by = "sort[" + filter_q_dict["sort_order"] + "]"
        filter_q_dict[sort_by] = "desc"
        
    extract_q = queries[2].removeprefix("Extract query: ")
    return search_q, filter_q_dict, extract_q


def clean_answers(answers):
    '''
    Clean answers
    Args:
        answers (dict): dictionary of answers per document
    Returns:
        answers_list (list): list of clean answers
    '''
    answers_list = [i["answer"] for i in answers.values()]
    clean_answers = []
    for answer in answers_list:
        if answer != " Unanswerable." and answer != " Unanswerable":
            clean_answers.append(answer)
    return clean_answers


def get_generated_answers(user_query, prompt_filename, path=ANSWER_PATH):
    '''
    Get generated answers from json file
    Args:
        user_query (str): query to search for documents
        prompt_filename (str): filename of prompt template
        path (str): path to answer directory
    Returns:
        answers_dict (dict): dictionary of answers per document
    '''
    with open(os.path.join(path, prompt_filename, user_query + ".json"), "r") as f:
        answers_dict = json.load(f)
    return answers_dict


def clean_generated_answers(gen_answers):
    for k, v in gen_answers.items():
        gen_answers[k] = v.replace("\n\n", "")
        gen_answers[k] = v.replace("\n", "")
        gen_answers[k] = gen_answers[k][1:] if gen_answers[k].startswith(" ") else gen_answers[k]
        gen_answers[k] = "Unanswerable" if "Unanswerable" in gen_answers[k] or "unanswerable" in gen_answers[k] else gen_answers[k]
        gen_answers[k] = "Yes" if gen_answers[k].startswith("Yes") else gen_answers[k]
        gen_answers[k] = "No" if gen_answers[k].startswith("No") else gen_answers[k]
    return gen_answers


def keep_relevant_docs(explanation, q_retrieved_psgs):
    '''
    This function extracts the relevant documents from the explanation and returns them.
    Code from Visconde repo
    Args:
        explanation: string containing the documents that are relevant to the answer
        q_retrieved_psgs: list containing the retrieved passages for the question (top 5)
    Returns:
        relevant_documents: list containing the relevant documents
    '''
    regex = r"\[Document \d+\]"
    matches = re.finditer(regex, explanation, re.MULTILINE)
    relevant_documents = []
    for match in matches:
        nums = re.findall(r'\b\d+\b', match.group())
        if len(nums) > 0:
            try:
                relevant_documents.append(q_retrieved_psgs[int(nums[0])-1])
            except:
                pass
    return relevant_documents
