import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import pymorphy2
import pymorphy3
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
import numpy as np
import string

# Загрузка необходимых ресурсов nltk
nltk.download('punkt_tab')


def lemmatize_and_stem(text):
    morph_p2 = pymorphy2.MorphAnalyzer()
    morph_p3 = pymorphy3.MorphAnalyzer()
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    morph_vocab = MorphVocab()

    stemmer = SnowballStemmer("russian")

    words = word_tokenize(text, language='russian')

    results = {
        "original": words,
        "pymorphy2": [morph_p2.parse(word)[0].normal_form for word in words],
        "pymorphy3": [morph_p3.parse(word)[0].normal_form for word in words],
        "natasha": []
    }

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        results["natasha"].append(token.lemma)

    results["stemmed"] = [stemmer.stem(word) for word in words]

    return results


def tokenize_ascii_rus(text):
    return [char for char in text if ord(char)>1000]


def vectorize_ascii_rus(text):
    return np.array([ord(char) for char in text if ord(char)>1000])

# Тест
sample_text = "Пример обработки текста в Python!"
lemmatized_stemmed = lemmatize_and_stem(sample_text)
tokenized = tokenize_ascii_rus(" ".join(lemmatized_stemmed["original"]))
vectorized = vectorize_ascii_rus(" ".join(lemmatized_stemmed["original"]))

print("Лемматизация и стемминг:", lemmatized_stemmed)
print("Токенизированный ASCII:", tokenized)
print("Векторизированный ASCII:", vectorized)
