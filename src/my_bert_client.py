from bert_serving.client import BertClient
from logzero import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS


def get_encoder_from_text(text=[]):
    bc = BertClient(ip='127.0.0.1', check_length=False)
    text_enc = bc.encode(text)
    # logger.info(f'get encoded vector {text_enc}')
    return text_enc


def get_similarity(text1=None, text2=None):
    text1_enc = get_encoder_from_text(text1)
    text2_enc = get_encoder_from_text(text2)
    similarity = cosine_similarity(text1_enc, text2_enc)
    return similarity
