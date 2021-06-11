import re
import time
import spacy
import numpy as np
import pandas as pd
from pandarallel import pandarallel

# Carregando a base de dados
df = pd.read_csv('data/Train50.csv', delimiter=';')
df = df.drop(['id', 'tweet_date', 'query_used'], axis=1)

# Carregando o modelo estatístico
nlp = spacy.load('pt_core_news_sm')

# Criando a função de pré-processamento do texto
def preprocessing(text):
    text = text.lower()

    # retira nome do usuário: @labdata
    text = re.sub(r"@[A-Za-z0-9$-_@.&+]+", '', text)

    # retira as URLs
    text = re.sub(r"https?://[A-Za-z0-9./]+", '', text)

    # retira espaços em branco extras no meio do texto, no começo e no fim
    text = re.sub(r" +", ' ', text).strip()

    # substituir emoticons por texto
    emoticons = {
        ':)': 'emocaopositiva',
        ':d': 'emocaopositiva',
        ':(': 'emocaonegativa'
    }
    for emoticon in emoticons:
        text = text.replace(emoticon, emoticons[emoticon])

    doc = nlp(text)

    # remove pontuacão, stop_word e dígitos
    # pega apenas o lemma da palavra
    lista = []
    for token in doc:
        if not token.is_punct and not token.is_stop and not token.like_num:
            lista.append(token.lemma_)

    # transforma a lista em um texto
    text_processed = ' '.join(lista)
    return text_processed


# Aplicando a função no dataframe completo, utilizando todos
# os cores da máquina com o pandarallel
pandarallel.initialize(progress_bar=True)

start = time.time()
df['tweet_text_cleaned'] = df['tweet_text'].parallel_apply(preprocessing)
end = time.time()
print(f"Tempo total de execução: {(end-start)/60:.2f} minutos")

# Salvando o dado tratado
df.to_csv("data/Train50.csv", index=False, sep=";")
