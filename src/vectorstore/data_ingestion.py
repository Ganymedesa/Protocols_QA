import os
import sys
import pandas as pd 
from src.logger import logging 
from dataclasses import dataclass
from langchain.document_loaders import PyPDFLoader
from exception import CustomException



@dataclass
class DataIngestionConfig:
    text_file_path = os.path.join('artifacts', 'text_file.pdf')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion stage ... Reading files")

        try:
            loader = PyPDFLoader('src/vectorstore/data/Pediatric_Hodgkin_Lymphoma_Protocol_1.pdf')
            pages = loader.load_and_split()

            logging.info('Finished Data Ingestion')
            return pages

        except Exception as e:
            raise CustomException(e, sys)

