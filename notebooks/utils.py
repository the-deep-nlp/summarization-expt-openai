from numpy import dot
from numpy.linalg import norm

def join_lst_elements(lst):
    """ Joins the list to form a string """
    return ". ".join(lst)


def remove_newline(generated_text):
    """ Removes the newline char """
    generated_text = generated_text.replace("\n", " ")
    return generated_text

def get_embeddings(
    openai_api_key,
    summary_text,
    model="text-embedding-ada-002"
):
    text = remove_newline(summary_text)
    return embedding_model.embed_query(text)

def get_cosine_similarity(first_v, second_v):
    """ Get the cosine similarity """
    return dot(first_v, second_v)/(norm(first_v)*norm(second_v))

