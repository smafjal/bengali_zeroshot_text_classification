from transformers import AutoTokenizer, AutoModelForSequenceClassification

pretrain_model_dir_bert = "neuralspace-reverie/indic-transformers-bn-bert"
pretrain_model_dir_robert = "neuralspace-reverie/indic-transformers-bn-roberta"
murli = "monsoon-nlp/muril-adapted-local"
distill_bert = "neuralspace-reverie/indic-transformers-bn-distilbert"
indo_bert = "ashwani-tanwar/Indo-Aryan-XLM-R-Base"
sagor_bert = "sagorsarker/bangla-bert-base"
electra = "monsoon-nlp/bangla-electra"


def read_sample(path):
    texts = []
    with open(path, 'r') as r:
        for line in r:
            line = line.strip()
            lines = line.split("#")
            texts.extend([l.strip() for l in lines])
    return texts


def main():
    # tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dir_robert)
    # tokenizer = AutoTokenizer.from_pretrained(murli)
    # tokenizer = AutoTokenizer.from_pretrained(indo_bert)
    # tokenizer = AutoTokenizer.from_pretrained(sagor_bert)
    tokenizer = AutoTokenizer.from_pretrained(electra)
    # tokenizer = AutoTokenizer.from_pretrained(distill_bert)
    # model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_dir_robert)

    dpath = "test.txt"
    texts = read_sample(dpath)
    print(texts[:5])

    for t in texts[:5]:
        print(tokenizer.tokenize(t))
    print(len(tokenizer.vocab))


if __name__ == '__main__':
    main()
