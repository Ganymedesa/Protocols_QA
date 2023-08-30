import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
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



# @dataclass
# class DataBaseCreationConfig():
#     processed_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

# export OPENAI_API_KEY=api_key



class DataBaseCreation():
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')

    MODEL_NAME = 'gpt-3.5-turbo'
    EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'

    def __init__(self, index_name):
        # self.data_processed_path = DataBaseCreationConfig()
        self.encoding = tiktoken.encoding_for_model(self.MODEL_NAME).name
        self.tokenizer = tiktoken.get_encoding(self.encoding)
        self.index_name = self.db_creation(index_name)
        self.embedding = self.model_embedding()


    def db_creation(self, index_name):
        pinecone.init(
        api_key=self.PINECONE_API_KEY,
        environment=self.PINECONE_ENVIRONMENT
    )

        if index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=index_name,
                metric='cosine',
                dimension=1536  # 1536 dim of text-embedding-ada-002
            )
            index = pinecone.Index(index_name)

            return index

    def model_embedding(self):
        embed = OpenAIEmbeddings(
            model=self.EMBEDDING_MODEL_NAME,
            openai_api_key=self.OPENAI_API_KEY
        ) 
        return embed

    def tiktoken_len(self, text):
        """calculate the number of tokens in the entire document """

        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def populating_db(self, embed, index, documents):

        batch_limit = len(documents)

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,  # number of tokens overlap between chunks
        length_function=self.tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
            )
        texts = []
        ids = []

        for i, record in enumerate(tqdm(documents)):

            record_texts = text_splitter.split_text(record.page_content)
            record_metadatas = [{"chunk": j, "text": text } for j, text in enumerate(record_texts)]
            texts.extend(record_texts)
            metadatas.extend(record_metadatas)
            # if we have reached the batch_limit we can add texts
            if len(texts) >= batch_limit:
                ids = [str(uuid4()) for _ in range(len(texts))]
                embeds = embed.embed_documents(texts)
                index.upsert(vectors=zip(ids, embeds, metadatas))
                texts = []
                metadatas = []
        return embed
        

    def initiate_objects(self, embed, index, OPENAI_API_KEY, text_field='text'):
        index = pinecone.Index(self.index_name)
        vectorstore = Pinecone(
            index, embed.embed_query, text_field
        )
        llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.2
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        

