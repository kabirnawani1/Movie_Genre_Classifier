import pyspark
import json
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit, array, concat_ws
from pyspark.sql.types import ArrayType, StringType,MapType
import re

import pymongo
from pymongo import MongoClient
from pymongo.server_api import ServerApi
#////////////////////////
def extractTags(string):
    return re.findall('"([\w\s]+)"', string)

def removeWord(listOfWords, word):
    return [x for x in listOfWords if x != word]

def replaceWord(listOfWords, word, replacements):
    return [x if x != word else replacements for x in listOfWords]

def process_data(df_input, toDelete, toReplace):
    
    # Drop rows with null values
    df_input = df_input.na.drop()
    # Define UDFs
    evaluate_udf = udf(lambda x:list(json.loads(x).values()),ArrayType(StringType()))
    extractTags_udf = udf(extractTags, ArrayType(StringType()))
    removeWord_udf = udf(removeWord, ArrayType(StringType()))
    replaceWord_udf = udf(replaceWord, ArrayType(StringType()))

    # # # Apply UDFs

    df_input = df_input.withColumn("movie_genres", extractTags_udf(df_input["movie_genres"]))
    df_input = df_input.withColumn("movie_languages", extractTags_udf(df_input["movie_languages"]))
    df_input = df_input.withColumn("movie_countries", extractTags_udf(df_input["movie_countries"]))
    # df_input.printSchema()
    
    # Apply wordRemover and wordReplacer
    for word in toDelete:
        df_input = df_input.withColumn("movie_genres", removeWord_udf(df_input["movie_genres"], lit(word)))
    for word, replacements in toReplace:
        df_input = df_input.withColumn("movie_genres", replaceWord_udf(df_input["movie_genres"], lit(word), array([lit(r) for r in replacements])))
    
    
    df_input = df_input.select('movie_name', 'movie_summary','movie_genres')
    
    return df_input

def ensure_collection_exists():
    client = MongoClient("mongodb+srv://moniregar:wxb0mHswOmoskZIS@cluster-groupproject.l9lrl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-groupproject",
                         server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Ensuring collection. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    db = client.groupproject
    if "movies_training_processed" in db.list_collection_names():
        db.movies_training_processed.drop()
    client.close()

def write_to_mongodb(row):
    client = MongoClient("mongodb+srv://moniregar:wxb0mHswOmoskZIS@cluster-groupproject.l9lrl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-groupproject",
                         server_api=ServerApi('1'))
    db = client.groupproject
    collection = db.movies_training_processed
    row_dict = row.asDict()
    collection.insert_one(row_dict)
    client.close()
    print('hola')

def main():
    ss = SparkSession \
    .builder \
    .appName("myApp") \
    .config("spark.mongodb.read.connection.uri", "mongodb+srv://moniregar:wxb0mHswOmoskZIS@cluster-groupproject.l9lrl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-groupproject") \
    .config("spark.mongodb.write.connection.uri", "mongodb+srv://moniregar:wxb0mHswOmoskZIS@cluster-groupproject.l9lrl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-groupproject") \
    .config("spark.mongodb.input.uri","mongodb+srv://moniregar:wxb0mHswOmoskZIS@cluster-groupproject.l9lrl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-groupproject")\
    .config("spark.mongodb.input.database","groupproject")\
    .config("spark.mongodb.input.collection","movies_training")\
    .config("spark.mongodb.output.uri", "mongodb+srv://moniregar:wxb0mHswOmoskZIS@cluster-groupproject.l9lrl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-groupproject/groupproject.movies_training_processed") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")\
    .getOrCreate()
    #.config("spark.driver.extraJavaOptions","-Xss200M")
    # .config("spark.mongodb.input.readPreference.name",=primaryPreferred
    df = ss.read.format("com.mongodb.spark.sql.DefaultSource").load()

    # Show the DataFrame schema and some sample data
    df.printSchema()
    toDelete = ['Absurdism', 'Airplanes and airports', 'Albino bias', 'Americana', 'Animal Picture', 'Animals', 'Anthology', 'Anthropology', 'Archaeology', 'Archives and records', 'Art film', 'Beach Film', 'Beach Party film', 'Bengali Cinema', 'Blaxploitation', 'Bollywood', 'British Empire Film', 'British New Wave', 'Buddy film', 'Business', 'Camp', 'Cavalry Film', 'Chase Movie', 'Chinese Movies', 'Christmas movie', 'Cold War', 'Coming of age', 'Computers', 'Cult', 'Cyberpunk', 'Dogme 95', 'Doomsday film', 'Early Black Cinema', 'Education', 'Educational', 'Environmental Science', 'Ensemble Film', 'Escape Film', 'Essay Film', 'Existentialism', 'Experimental film', 'Exploitation', 'Expressionism', 'Fan film', 'Feature film', 'Female buddy film', 'Feminist Film', 'Fictional film', 'Filipino', 'Filipino Movies', 'Film', 'Film adaptation', 'Filmed Play', 'Foreign legion', 'Giallo', 'Goat gland', 'Gothic Film', 'Gross out', 'Hagiography', 'Holiday Film', 'Indie', 'Japanese Movies', 'Journalism', 'Jungle Film', 'Juvenile Delinquency Film', 'Kafkaesque', 'Kitchen sink realism', 'Latino', 'Libraries and librarians', 'Linguistics', 'Live action', 'Media Studies', 'Medical fiction', 'Mondo film', 'Movie serial', 'Mumblecore', 'Nature', 'New Hollywood', 'News', 'Northern', 'Nuclear warfare', 'Parkour in popular culture', 'Patriotic film', 'Pinku eiga', 'Plague', 'Point of view shot', 'Prison', 'Private military company', 'Propaganda film', 'Reboot', 'Remake', 'Religious Film', 'Roadshow theatrical release', 'School story', 'Sexploitation', 'Sponsored film', 'Short Film', 'Singing cowboy', 'Slice of life story', 'Social issues', 'Social problem film', 'Sponsored film', 'Star vehicle', 'Statutory rape', 'Steampunk', 'Stoner film', 'Superhero', 'Superhero movie', 'Surrealism', 'Sword and Sandal', 'Sword and sorcery', 'Sword and sorcery films', 'Television movie', 'The Netherlands in World War II', 'Tragedy', 'Travel', 'World cinema', 'Wuxia', 'Z movie']
    toReplace = [('Acid western', ['Western']), ('Action Comedy', ['Action', 'Comedy']), ('Action Thrillers', ['Action', 'Thriller']), ('Addiction Drama', ['Drama']), ('Adventure Comedy',['Adventure', 'Comedy']), ('Alien Film', ['Creature Film', 'Science Fiction']), ('Alien invasion', ['Creature Film', 'Science Fiction']), ('Animated Musical', ['Animation']), ('Animated cartoon', ['Animation']), ('Anime', ['Animation']), ('Auto racing', ['Sports']), ('Backstage Musical', ['Musical']), ('Baseball', ['Sports']), ('Biker Film', ['Road movie']), ('Biographical film', ['Biography']), ('Black comedy', ['Comedy']), ('Boxing', ['Sports']), ('Breakdance', ['Dance']), ('Buddy cop', ['Crime']), ('Caper story', ['Crime', 'Comedy']), ('Chick flick', ['Romance']), ('Childhood Drama',['Drama']), ('Christian film', ['Religious Film']), ('Clay animation', ['Animation']),
             ('Combat Films', ['Action']), ('Comdedy',['Comedy']), ('Comedy Thriller', ['Comedy', 'Thriller']), ('Comedy Western', ['Comedy', 'Western']), ('Comedy film', ['Comedy']), ('Comedy horror', ['Comedy', 'Horror']), ('Comedy of Errors', ['Comedy']), ('Comedy of manners', ['Comedy']), ('Computer Animation', ['Animation']), ('Concert film', ['Music']), ('Conspiracy fiction', ['Thriller']), ('Costume Adventure', ['Adventure']), ('Costume Horror', ['Horror']), ('Costume drama', ['Drama']), ('Courtroom Comedy',['Courtroom', 'Comedy']), ('Courtroom Drama',['Courtroom', 'Drama']), ('Creature Film', ['Monster']), ('Crime Comedy', ['Crime', 'Comedy']), ('Crime Drama', ['Crime', 'Drama']), ('Crime Fiction', ['Crime']), ('Crime Thriller', ['Crime', 'Thriller']), ('Demonic child', ['Horror']), ('Detective fiction', ['Detective']), ('Docudrama', ['Drama']), ('Domestic Comedy', ['Comedy']), ('Ealing Comedies', ['Comedy']), ('Epic Western', ['Epic', 'Western']), ('Erotic Drama', ['Adult', 'Drama']), ('Erotic thriller', ['Adult', 'Thriller']), ('Erotica', ['Adult']), ('Extreme Sports', ['Sports']), ('Family Drama', ['Family Film', 'Drama']),
             ('Fairy Tale', ['Fantasy']), ('Fairy tale', ['Fantasy']), ('Fantasy Adventure', ['Fantasy', 'Adventure']), ('Fantasy Comedy', ['Fantasy', 'Comedy']), ('Fantasy Drama', ['Fantasy', 'Drama']), ('Future noir', ['Film noir']), ('Gangster Film', ['Crime']), ('Gay', ['LGBT']), ('Gay Interest', ['LGBT']), ('Gay Themed', ['LGBT']), ('Gay pornography', ['LGBT', 'Adult']), ('Gender Issues', ['LGBT']), ('Glamorized Spy Film', ['Spy']),
             ('Gulf War', ['War film']), ('Haunted House Film', ['Horror']), ('Hardcore pornography', ['Adult']), ('Heavenly Comedy', ['Comedy']), ('Heist', ['Crime']), ('Hip hop movies', ['Music']), ('Historical Documentaries', ['History', 'Documentary']), ('Historical Epic', ['History']), ('Historical drama', ['History']), ('Historical Drama', ['History', 'Drama']), ('Historical fiction', ['History']), ('Homoeroticism', ['Adult', 'LGBT']), ('Horror Comedy', ['Horror', 'Comedy']), ('Horse racing', ['Sport']), ('Humour', ['Comedy']), ('Hybrid Western', ['Western']), ('Indian Western', ['Western']), ('Inspirational Drama', ['Drama']), ('Instrumental Music', ['Music']), ('Interpersonal Relationships', ['Drama']), ('Jukebox musical', ['Musical']), ('Legal drama', ['Courtroom']), ('Marriage Drama', ['Drama']), ('Master Criminal Films', ['Crime']), ('Media Satire', ['Comedy']),
             ('Melodrama', ['Drama']), ('Mockumentary', ['Comedy']), ('Monster movie', ['Monster']), ('Movies About Gladiators', ['History', 'Action']), ('Musical Drama', ['Musical', 'Drama']), ('Musical comedy', ['Musical', 'Comedy']), ('Mythological Fantasy',['Fantasy']), ('Natural disaster', ['Disaster']), ('Natural horror films', ['Horror']), ('Ninja movie', ['Martial Arts Film']), ('Operetta', ['Musical']), ('Outlaw', ['Crime']), ('Outlaw biker film', ['Crime', 'Road movie']), ('Parody', ['Comedy']),('Period Horror', ['Period', 'Horror']), ('Period piece', ['Period']), ('Political cinema', ['Politics']), ('Political drama', ['Politics', 'Drama']), ('Political satire', ['Politics', 'Comedy']), ('Political thriller', ['Politics', 'Thriller']), ('Pornographic movie', ['Adult']), ('Pornography', ['Adult']), ('Prison', ['Crime']), ('Prison escape', ['Crime']), ('Prison film', ['Crime']), ('Psychological horror', ['Horror']), ('Psychological thriller', ['Thriller']),
             ('Punk rock', ['Music']), ('Race movie', ['Sports']), ('Revisionist Fairy Tale', ['Fantasy']), ('Revisionist Western', ['Western']), ('Rockumentary', ['Documentary', 'Music']), ('Romance Film', ['Romance']), ('Romantic Film', ['Romance']), ('Romantic comedy', ['Romance', 'Comedy']), ('Romantic drama', ['Romance', 'Drama']), ('Romantic fantasy', ['Romance', 'Fantasy']), ('Samurai cinema', ['Martial Arts Film']), ('Satire', ['Comedy']), ('Sci Fi Pictures original films', ['Science Fiction']), ('Science fiction Western', ['Science Fiction', 'Western']), ('Screwball comedy', ['Comedy']), ('Sex comedy', ['Comedy']), ('Slapstick', ['Comedy']), ('Slasher', ['Horror']), ('Softcore Porn', ['Adult']), ('Space opera', ['Science Fiction', 'Musical']), ('Space western', ['Science Fiction', 'Western']), ('Spaghetti Western', ['Western']), ('Spaghetti western', ['Western']), ('Splatter film', ['Horror']), ('Sport', ['Sports']), ('Stop motion', ['Animation']), ('Supermarionation', ['Animation']), ('Swashbuckler films', ['Adventure']),
             ('Therimin music', ['Music']), ('Time travel', ['Science Fiction']), ('Tragicomedy', ['Comedy', 'Tragedy']), ('Vampire movies', ['Horror']), ('War effort', ['War film']), ('Werewolf fiction', ['Monster']), ('Whodunit', ['Detective']), ('Women in prison films', ['Prison']), ('World History', ['History']), ('Workplace Comedy', ['Comedy']), ('Zombie Film', ['Monster'])]
    df_processed = process_data(df, toDelete, toReplace)
    df_processed.show(5)
    # ensure_collection_exists()
    # df_processed.foreach(write_to_mongodb)
    df_processed.write.format("com.mongodb.spark.sql.DefaultSource")\
                        .option("database","groupproject")\
                        .option("collection", "movies_training_processed")\
                        .mode("append").save()

    ss.stop()
if __name__ == "__main__":
    main()

