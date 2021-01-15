import os
import pandas as pd
from ast import literal_eval
from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM




download_model(model='bert-squad_1.1', dir='./models')


df = pdf_converter(directory_path='tests/examen-acceso')
df.head()

cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib', max_df=1.0)

# Fit Retriever to documents
cdqa_pipeline.fit_retriever(df=df)

query = 'si el cliente no paga, me puedo quedar la documentación del caso y no devolver'
prediction = cdqa_pipeline.predict(query)


print('query: {}'.format(query))
print('answer: {}'.format(prediction[0]))
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))