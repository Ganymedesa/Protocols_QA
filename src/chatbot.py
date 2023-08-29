import os
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.auto import tqdm
import hashlib
import tiktoken
from tqdm.auto import tqdm
from uuid import uuid4
from langchain.vectorstores import Pinecone




print('hello')