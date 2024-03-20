from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage
import pymongo
from pymongo import MongoClient, InsertOne
from pymongo.server_api import ServerApi

import json
import numpy as np
import pandas as pd
import re
from io import BytesIO
import os
import string
import emoji
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')


pd.set_option("display.max_columns", None)


class TextPreprocessing:

    contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                           "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                           "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am",
                           "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                           "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                           "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                           "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                           "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                           "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                           "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                           "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
                           "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                           "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                           "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                           "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                           "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                           "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                           "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                           "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s': 'america', 'e.g': 'for example'}

    punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
             '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
             '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
             '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
             '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                     "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
                     'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!': ' '}

    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                    'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                    'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                    'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                    'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum',
                    'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                    'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                    'demonetisation': 'demonetization'}

    @staticmethod
    def clean_text(text):
        '''Clean emoji, Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.'''
        text = emoji.demojize(text)
        text = re.sub(r'\:(.*?)\:', '', text)
        text = str(text).lower()  # Making Text Lowercase
        text = re.sub('\[.*?\]', '', text)
        # The next 2 lines remove html text
        text = BeautifulSoup(text, 'lxml').get_text()
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
        text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
        return text

    @staticmethod
    def clean_contractions(text, mapping):
        '''Clean contraction using contraction mapping'''
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        for word in mapping.keys():
            if ""+word+"" in text:
                text = text.replace(""+word+"", ""+mapping[word]+"")
        # Remove Punctuations
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        text = re.sub(r"([?.!,¿])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        return text

    @staticmethod
    def clean_special_chars(text, punct, mapping):
        '''Cleans special characters present(if any)'''
        for p in mapping:
            text = text.replace(p, mapping[p])

        for p in punct:
            text = text.replace(p, f' {p} ')

        specials = {'\u200b': ' ', '…': ' ... ',
                    '\ufeff': '', 'करना': '', 'है': ''}
        for s in specials:
            text = text.replace(s, specials[s])

        return text

    @staticmethod
    def correct_spelling(x, dic):
        '''Corrects common spelling errors'''
        for word in dic.keys():
            x = x.replace(word, dic[word])
        return x

    @staticmethod
    def remove_space(text):
        '''Removes awkward spaces'''
        text = text.strip()
        text = text.split()
        return " ".join(text)

    @staticmethod
    def pipeline(text):
        '''Cleaning and parsing the text.'''
        text = TextPreprocessing.clean_text(text)
        text = TextPreprocessing.clean_contractions(
            text, TextPreprocessing.contraction_mapping)
        text = TextPreprocessing.clean_special_chars(
            text, TextPreprocessing.punct, TextPreprocessing.punct_mapping)
        text = TextPreprocessing.correct_spelling(
            text, TextPreprocessing.mispell_dict)
        text = TextPreprocessing.remove_space(text)
        return text


class BERTTestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.max_len = max_len
        self.text = df.summary
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained('roberta-base')
        self.fc = torch.nn.Linear(768, 10)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.fc(features)
        return output


class ModelUtils:
    @staticmethod
    def load_model(blob):
        model = BERTClass()
        bin_blob = blob.download_as_bytes()
        try:
            with open('model.bin','wb') as f:
                f.write(bin_blob)
            model.load_state_dict(torch.load(
                'model.bin', map_location=torch.device('cpu')))
        except:
            state_dict = torch.load(BytesIO(bin_blob), map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        device = 'cpu'
        return model, device

    @staticmethod
    def validation(pred_loader, model):
        os.environ["TOKENIZERS_PARALLELISM"]="false"
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(pred_loader, 0):
                ids = data['ids']
                mask = data['mask']
                token_type_ids = data['token_type_ids']
                outputs = model(ids, mask, token_type_ids)
                fin_outputs.extend(torch.sigmoid(
                    outputs).cpu().detach().numpy().tolist())
                
        return fin_outputs

    @staticmethod
    def get_pred_genres(validation, pred_loader, model, device, threshold=0.5):
        outputs = validation(pred_loader, model)
        outputs = np.array(outputs) >= threshold
        genres = ["Drama", "Comedy", "Romance", "Thriller", "Action",
                  "Crime", "Horror", "Family Film", "Adventure", "Animation"]
        values_array = np.array(outputs)
        pred_genres = [np.array(genres)[value_row]
                       for value_row in values_array]
        return pred_genres


def load_model_and_tokenizer():
    blob_name = "dags/model/model.bin"  # Replace with your actual file path
    bucket_name = 'us-west1-msds-dds-project-53a89e6e-bucket'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    MAX_LEN = 200
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model, device = ModelUtils.load_model(blob)
    return model, device, tokenizer, MAX_LEN


def predict_genre(df, model, device, tokenizer, MAX_LEN):
    TRAIN_BATCH_SIZE = 64
    df['summary'] = df['summary'].apply(TextPreprocessing.pipeline)
    pred_data = BERTTestDataset(df, tokenizer, MAX_LEN)
    pred_loader = DataLoader(pred_data, batch_size=TRAIN_BATCH_SIZE,
                             num_workers=3, shuffle=False, pin_memory=True)
    pred_genres = ModelUtils.get_pred_genres(
        ModelUtils.validation, pred_loader, model, device)
    return pred_genres


def model_prediction(data):
    print('statement_1')
    text = f"""Cassandra Webb develops the power to see the future. 
                Forced to confront revelations about her past, 
                she forges a relationship with three young women bound 
                for powerful destinies, if they can all survive a deadly present
            """
    data = json.load(data)
    plot = data.get('Plot')
    df = pd.DataFrame({'summary': [plot]})
    model, device, tokenizer, MAX_LEN = load_model_and_tokenizer()
    preds = predict_genre(df, model, device, tokenizer, MAX_LEN)
    predicted_genres = [arr.tolist() for arr in preds]
    print(predicted_genres)
    return 1


def write_to_mongodb(data,**kwargs):
    ti = kwargs['ti']
    # Connect to MongoDB Atlas
    client = MongoClient("mongodb+srv://moniregar:wxb0mHswOmoskZIS@cluster-groupproject.l9lrl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-groupproject",
                         server_api=ServerApi('1'))
    db = client.groupproject
    collection = db.movies_training_processed

    data = json.load(data)
    name = data.get('Title')
    pred = ti.xcom_pull(task_ids=['interact_with_model'])
    plot = data.get('Plot')
    data = {"movie_name": name, "plot":plot, "pred_genre": pred}
    collection.insert_one(data)
    client.close()

default_args = {
    'start_date': datetime(2024, 1, 1),
    "retries": 2,
}

# Define your DAG
dag = DAG('model_dag_final', 
         default_args=default_args,
         start_date=datetime(2024, 1, 1),
         schedule_interval=None,)

read_from_gcs = BashOperator(
    task_id='read_files',
    bash_command='gsutil cat gs://us-west1-msds-dds-project-53a89e6e-bucket/data/new_releases/new_movie.json',
    dag=dag)

task1 = PythonOperator(
    task_id='interact_with_model',
    python_callable=model_prediction,
    op_args=[read_from_gcs.output],
    dag=dag)

write_to_mongodb_task = PythonOperator(
    task_id='write_to_mongodb',
    python_callable=write_to_mongodb,
    op_args = [read_from_gcs.output]
    dag=dag)


# Add more tasks as needed
read_from_gcs >> task1 >> write_to_mongodb_task