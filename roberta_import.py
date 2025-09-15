from transformers import RobertaTokenizer
RobertaTokenizer.from_pretrained("roberta-base").save_pretrained("./roberta-base")