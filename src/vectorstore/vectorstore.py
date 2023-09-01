import os
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# from src.components.data_ingestion import DataIngestion

from langchain.vectorstores import Pinecone
import pinecone
from tqdm.auto import tqdm
from uuid import uuid4
from src.logger import logging 
from tqdm.auto import tqdm
import hashlib
from dataclasses import dataclass
from src.utils import save_object
import tiktoken
import os
from exception import CustomException



# @dataclass
# class DataBaseCreationConfig():
#     processed_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

# export OPENAI_API_KEY=api_key



class DataBaseCreation():
    logging.info("Database object creation started ...")
    logging.info("Initializing API keys and models")

    # MODEL_NAME = 'gpt-3.5-turbo'
    # EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'

    def __init__(self, index_name):
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        self.PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        self.PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')

        self.index_name = index_name 
        self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo').name
        self.tokenizer = tiktoken.get_encoding(self.encoding)

        self.embed = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=self.OPENAI_API_KEY
        ) 
        

    def initialize_vec_db(self):
        try:
            logging.info("vectorestore creation method started")

            pinecone.init(
            api_key=self.PINECONE_API_KEY,
            environment=self.PINECONE_ENVIRONMENT
        )
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    metric='cosine',
                    dimension=1536  # 1536 dim of text-embedding-ada-002
                )
                self.new_index = pinecone.Index(self.index_name)
            else:
                self.new_index = pinecone.Index(self.index_name)
            logging.info("Finished vectorestore")
        
        except Exception as e:
            raise CustomException(e, sys)
                

    def tiktoken_len(self, text):
        """calculate the number of tokens in the entire document """

        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def populating_db(self, documents):
        try:
            logging.info("Populating database method started")
            batch_limit = len(documents)

            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,  # number of tokens overlap between chunks
            length_function=self.tiktoken_len,
            separators=['\n\n', '\n', ' ', '']
                )
            texts = []
            metadatas = []

            for record in (tqdm(documents)):

                record_texts = text_splitter.split_text(record.page_content)
                record_metadatas = [{"chunk": j, "text": text } for j, text in enumerate(record_texts)]
                texts.extend(record_texts)
                metadatas.extend(record_metadatas)
                # if we have reached the batch_limit we can add texts
                if len(texts) >= batch_limit:
                    ids = [str(uuid4()) for _ in range(len(texts))]
                    embeds = self.embed.embed_documents(texts)
                    self.new_index.upsert(vectors=zip(ids, embeds, metadatas))
                    texts = []
                    metadatas = []

            text_field='text'
            index = pinecone.Index(self.index_name)
            vectorstore = Pinecone(
                index, self.embed.embed_query, text_field
            )
            logging.info("Finished populating database")
            return vectorstore
        
        except Exception as e:
            raise CustomException(e, sys)
        

    
        

