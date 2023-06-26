import argparse
import pipeline.utils as utils
import os
import json
import pandas as pd

from pipeline.get_collection import call_api, save_collection
from pipeline.gen_answers import get_answer_per_paper
from pipeline.constants import API_BASE_URL
from tabulate import tabulate, SEPARATING_LINE

def get_search_filter_extract_query(query, prompt_filename="search_filter_extract_query_prompt"):
    prompt = utils.construct_prompt(
        filename=f"{prompt_filename}.txt",
        prompt_params_dict={
            "current_dt": utils.get_current_date(),
            "input_query": query,
            "dt_5_years_ago": utils.get_date_five_years_ago(),
        }
    )

    print("Generating Search, Filter and Extract queries with text-davinci-003...")
    response = utils.get_model_response(prompt, "text-davinci-003")["choices"][0]["text"]

    search_q, filter_q_dict, extract_q = utils.parse_search_filter_extract_response(response)
    print(f"Search query: {search_q}")
    print(f"Filter query: {filter_q_dict}")
    print(f"Extract query: {extract_q}")
    return search_q, filter_q_dict, extract_q


def parse_args():
    parser = argparse.ArgumentParser()
    # either user_query or query_file is required
    group = parser.add_mutually_exclusive_group(required=True)
    # read query from command line
    group.add_argument("--query", type=str, help="Query to search for documents")
    # read query from file
    group.add_argument("--query_file", type=argparse.FileType("r"), help="File containing queries to search for documents")
    # define url of API
    parser.add_argument("--url", type=str, default=API_BASE_URL, help="URL of API to call to get collection of documents")
    # set number of documents to retrieve
    parser.add_argument("--page_size", type=int, default=50, help="Number of documents to retrieve")
    # model to use
    parser.add_argument("--model", type=str, choices=["text-davinci-003"], default="text-davinci-003")
    # choose wheather to aggregate results
    parser.add_argument("--aggregate", default=False, action=argparse.BooleanOptionalAction, help="Aggregate results into clusters with BERTopic")

    # choose question answering prompt to use
    parser.add_argument(
        "--prompt_filename", 
        default="qasper_fewshot_prompt",
        type=str, 
        choices=[
            "qasper_zeroshot_prompt", 
            "qasper_fewshot_prompt",
            "qasper_cot_prompt", 
        ],
        help="Filename of QA prompt to use (to extract answer per document)."
    )
    # choose between knn or keyword retrieval
    parser.add_argument(
        "--retrieval_method", 
        type=str, 
        default="knn", 
        choices=["knn", "keyword"],
        help="Retrieval method to use"
    )

    # n_neighbors=4, n_components=6, min_cluster_size=3
    parser.add_argument("--n_neighbors", type=int, default=4, help="Number of neighbors to consider for UMAP")
    parser.add_argument("--n_components", type=int, default=6, help="Number of dimensions for UMAP")
    parser.add_argument("--min_cluster_size", type=int, default=3, help="Minimum number of documents in a cluster")

    return parser.parse_args()


def main():
    args = parse_args()

    # check if we already have the answers for the query
    if os.path.exists(os.path.join("data", "answers", f"{args.prompt_filename}", f"{args.query}.json")):
        print("Answers already exist for this query.")
        answers_dict = utils.get_generated_answers(args.query, args.prompt_filename)
        answers = utils.clean_answers(answers_dict)

        if args.aggregate:
            from pipeline.topic_modelling import save_clusters, generate_topic_labels, initialize_topic_model
            print(f"Saving clusters in clusters/{args.query}.json")
            # Initialize and fit topic model
            print("Fitting topic model to collection...")
            topic_model, embedding_model = initialize_topic_model(args.n_neighbors, args.n_components, args.min_cluster_size)
            topic_model.fit_transform(answers)

            # Generate topic labels
            print("Generating topic labels...")
            topic_model = generate_topic_labels(topic_model, args.query)

            print(topic_model.get_topic_info())
            save_clusters(answers=answers, embedding_model=embedding_model, topic_model=topic_model, query=args.query, prompt_filename=args.prompt_filename)
            return

        answers_df = pd.DataFrame(answers_dict)
        titles = answers_df.T['title'].tolist()
        answers = answers_df.T['answer'].tolist()
        tabul = [[titles[i], answers[i]] for i in range(len(titles))]
        # add seperating lines between each answer
        for i in range(len(tabul)):
            tabul.insert(2*i+1, [SEPARATING_LINE, SEPARATING_LINE])
        print(tabulate(tabul, headers=["Title", "Answer"], tablefmt='psql', showindex="never", maxcolwidths=[30, 70]))
        return

    # Step 1: From user input generate filter and extraction query
    search_q, filter_q_dict, extract_q = get_search_filter_extract_query(args.query)

    # Step 2: Call API to get collection of documents using the filter query
    filters_dict = {
        "query_string": search_q, 
        "retrieval_method": args.retrieval_method,
        "document_types": "document",
        "with_code": "false",
        "page_size": args.page_size,
        "sources" : "arXiv",
        **filter_q_dict
    }

    print("Calling API to get collection of documents...")
    hits, url = call_api(filters_dict, args.url)

    # Step 3: Save collection of documents to json file
    print("Saving collection of documents to json file under data/collections...")
    collection = save_collection(url, hits, args.query, search_q, extract_q, filter_q_dict, args)

    # Step 4: Extract an answer per document using the extraction query
    print(f"Generating answer per document...")

    answers_dict = get_answer_per_paper(collection, args.prompt_filename, args.model)
    
    # save answers to json file
    print(f"Saving answers to data/answers/{args.prompt_filename}...")
    if not os.path.exists(os.path.join("data", "answers", f"{args.prompt_filename}")):
        os.makedirs(os.path.join("data", "answers", f"{args.prompt_filename}"))
    with open(os.path.join("data", "answers", f"{args.prompt_filename}", f"{args.query}.json"), "w") as f:
        json.dump(answers_dict, f, indent=4)

    # Step 5: Tabulate or Aggregate answers
    answers = utils.clean_answers(answers_dict)
    if args.aggregate:
        from pipeline.topic_modelling import save_clusters, generate_topic_labels, initialize_topic_model
        # Aggregate answers into topics using BERTopic
        print("Clustering answers into topics using BERTopic...")
        topic_model, embedding_model = initialize_topic_model(args.n_neighbors, args.n_components, args.min_cluster_size)
        topic_model.fit_transform(answers)

        # Generate topic labels
        print("Generating topic labels...")
        topic_model = generate_topic_labels(topic_model, args.query)

        print(topic_model.get_topic_info())
        save_clusters(answers=answers, embedding_model=embedding_model, topic_model=topic_model, query=args.query, prompt_filename=args.prompt_filename)
    
    answers_df = pd.DataFrame(answers_dict)
    titles = answers_df.T['title'].tolist()
    answers = answers_df.T['answer'].tolist()
    tabul = [[titles[i], answers[i]] for i in range(len(titles))]
    for i in range(len(tabul)):
        tabul.insert(2*i+1, [SEPARATING_LINE, SEPARATING_LINE])
    print(tabulate(tabul, headers=["Title", "Answer"], tablefmt='psql', showindex="never", maxcolwidths=[30, 70]))

if __name__ == "__main__":
    main()