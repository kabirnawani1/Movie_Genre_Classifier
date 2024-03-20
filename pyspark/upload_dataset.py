import pymongo
from pymongo import MongoClient, InsertOne
from pymongo.server_api import ServerApi
import pandas as pd

client = MongoClient("mongodb+srv://moniregar:wxb0mHswOmoskZIS@cluster-groupproject.l9lrl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-groupproject",
                             server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.groupproject
collection = db.movies_training

data_path = "/Users/igarciamontoya/airflow/dags/data/movie.metadata.tsv"
df_movies = pd.read_csv(data_path, delimiter='\t', names=['wikipedia_movieId',	
                                                    'freebase_movieId',
                                                    'movie_name',
                                                    'movie_release_date',
                                                    'movie_box_revenue',
                                                    'movie_runtime',
                                                    'movie_languages',
                                                    'movie_countries',
                                                    'movie_genres'])

summaries = []
with open("/Users/igarciamontoya/airflow/dags/data/plot_summaries.txt", 'r') as file:
    for line in file:
        identifier, text = line.strip().split('\t', 1)
        summaries.append((int(identifier), text))
df_summaries = pd.DataFrame(summaries, columns=['wikipedia_movieId', 'movie_summary'])

final_collection = pd.merge(df_movies,df_summaries,how='left',on='wikipedia_movieId')
for index, row in final_collection.iterrows():
    row_dict = row.to_dict()
    try:
        collection.insert_one(row_dict)
    except Exception: continue

client.close()
