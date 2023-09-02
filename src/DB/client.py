from dotenv import load_dotenv, find_dotenv
import os
from pymongo import MongoClient

load_dotenv(find_dotenv())
password = os.getenv('MONGODB_PWD')
connection_string = f'mongodb+srv://hattansa:{password}@chatbot.sxjztze.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(connection_string)
# cahtbot_db is the database name, and chatbot is the collection name
chatbot_db = client.chatbot

def insert_message(message, response):
    collection = chatbot_db.chatbot
    doc = {
        "message": message,
        "response": response
    }
    inserted_id = collection.insert_one(doc).inserted_id