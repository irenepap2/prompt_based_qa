import os

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from pipeline.utils import construct_prompt, get_model_response

def initialize_topic_model(n_neighbors=4, n_components=6, min_cluster_size=3, random_state=42):
    '''
    Initialize topic model
    Args:
        n_neighbors (int): number of neighbors to consider for each point in the UMAP embedding
        n_components (int): number of dimensions in the UMAP embedding
        min_cluster_size (int): minimum size of clusters in the HDBSCAN model
        random_state (int): random state
    Returns:
        topic_model (BERTopic): topic model
        embedding_model (SentenceTransformer): sentence transformer model
    '''

    # Step 1 - Extract embeddings
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(
        n_neighbors=n_neighbors, 
        n_components=n_components,
        metric='cosine', 
        random_state=random_state
    )

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, 
        metric='euclidean', 
        cluster_selection_method='eom', 
        prediction_data=True
    )

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english")

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()

    topic_model = BERTopic(
        embedding_model=embedding_model,    
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,  
        diversity=0.5,  
        # same as min_cluster_size               
        # min_topic_size=3,
    )

    return topic_model, embedding_model


def save_clusters(answers, embedding_model, topic_model, query, prompt_filename, reduce_embeddings=True, random_state=42):
    '''
    Save clusters under clusters/{query}.html
    Args:
        answers (list): list of answers
        embedding_model (SentenceTransformer): embedding model
        topic_model (BERTopic): topic model
        query (str): user query to save cluster as
        random_state (int): random state (seed for UMAP reduction)
    Returns:
        None
    '''
    embeddings = embedding_model.encode(answers, show_progress_bar=False)
    if reduce_embeddings:
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=random_state).fit_transform(embeddings)
        fig = topic_model.visualize_documents(answers, reduced_embeddings=reduced_embeddings, custom_labels=True)
    else:
        fig = topic_model.visualize_documents(answers, embeddings=embeddings, custom_labels=True)
    
    # Check if clusters and clusters/{prompt_filename} directory exists
    if not os.path.exists("clusters"):
        os.mkdir("clusters")
    if not os.path.exists(f"clusters/{prompt_filename}"):
        os.mkdir(f"clusters/{prompt_filename}")

    fig.update_layout(
        title={
            'text': f"<b>{query}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        }
    )
    fig.write_html(f"clusters/{prompt_filename}/{query}.html")


def parse_cluster_labels_response(response):
    cluster_labels = {} 
    clusters = response.split("\n")
    for cluster in clusters:
        label, name = cluster.split(":")
        label_id = int(label[-1])
        cluster_labels[label_id] = name[1:]
    return cluster_labels


def generate_topic_labels(topic_model, query):
    '''
    Generate topic labels using GPT-3
    Args:
        topic_model (BERTopic): topic model
        query (str): user query
    Returns:
        topic_model (BERTopic): topic model with updated labels
    '''

    prompt = construct_prompt(
    "create_labels_prompt.txt", 
    prompt_params_dict = {       
        "query": query,
        "clusters" : topic_model.representative_docs_
        }
    )

    response = get_model_response(prompt, "text-davinci-003")["choices"][0]["text"]
    print(response)
    cluster_labels = parse_cluster_labels_response(response)
    topic_model.set_topic_labels(cluster_labels)
    return topic_model

