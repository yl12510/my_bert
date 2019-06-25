from logzero import logger


def test_get_similarity():
    from src.my_bert_client import get_similarity
    text1 = [
        'In an interview with BBC political editor Laura Kuenssberg, Mr Hunt said he and Mr Johnson both wanted to change the Brexit deal negotiated by Mrs May - and their aspirations for the substance of a new deal were similar.']
    text2 = [
        'En una entrevista con la editora política de la BBC, Laura Kuenssberg, el Sr. Hunt dijo que tanto él como el Sr. Johnson querían cambiar el acuerdo Brexit negociado por la Sra. May, y que sus aspiraciones por el contenido de un nuevo acuerdo eran similares.']

    similarity = get_similarity(text1, text2)
    logger.info(f'similarity: {similarity}')
    assert similarity > 0.5
