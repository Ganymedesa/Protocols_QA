from flask import Flask, request
import openai
from src.DB.client import insert_message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv, find_dotenv
import os
from pymongo import MongoClient

from langchain.vectorstores import Pinecone

load_dotenv(find_dotenv())

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
MONGODB_PWD = os.getenv('MONGODB_PWD')

def initiate_objects():

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )
    text_field = "text"
    index_name = 'protocol-retrieval-augmentation'
    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    index = pinecone.Index(index_name)
    vectorstore = Pinecone(
        index, embed, text_field
)

    connection_string = f'mongodb+srv://hattansa:{MONGODB_PWD}@chatbot.sxjztze.mongodb.net/?retryWrites=true&w=majority'
    client = MongoClient(connection_string)
    # cahtbot_db is the database name, and chatbot is the collection name
    chatbot_db = client.chatbot
    collection = chatbot_db.chatbot


    template = """You are a helpful assistant that answer doctors' questions
        about Pediatric Hodgkin Lymphoma only. You must not answer any other questions.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        keep the answer as concise as possible
        {context}
        # Question: {question}
        # Helpful Answer
        """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.2
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        retriever=vectorstore.as_retriever()
    )
    return qa, vectorstore, collection
        


def insert_message(message, response, collection):
    # collection = chatbot_db.chatbot
    doc = {
        "message": message,
        "response": response
    }
    collection.insert_one(doc).inserted_id

def generate_answer(query):
    qa_obj, vec_db, chatbot_collec = initiate_objects()

    vec_db.similarity_search(
    query,  
    k=3  
    )

    answer = qa_obj.run(query)
    insert_message(query, answer, chatbot_collec)
    return answer

@app.route('/chatgpt', methods=['POST'])
def chatgpt():

    incoming_que = request.values.get('Body', '').lower()
    # print("Question: ", incoming_que)
    # Generate the answer using GPT-3
    # db_obj.initiate_objects(vec_db, incoming_que)
    answer = generate_answer(incoming_que)
    # insert_message(incoming_que, answer, )
    # print("BOT Answer: ", answer)
    bot_resp = MessagingResponse()
    msg = bot_resp.message()
    msg.body(answer)

    return str(bot_resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)
    