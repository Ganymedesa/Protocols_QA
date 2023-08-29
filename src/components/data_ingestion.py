import os
import pandas as pd 
from src.logger import logging 
from dataclasses import dataclass
from langchain.document_loaders import PyPDFLoader



@dataclass
class DataIngestionConfig:
    text_file_path = os.path.join('artifacts', 'text_file.pdf')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion stage ... Reading files")

        try:
            loader = PyPDFLoader("../data/Pediatric_Hodgkin_Lymphoma_Protocol_1.pdf")
            pages = loader.load_and_split()

            os.makedirs(os.path.dirname(self.ingestion_config.text_file_path), exist_ok=True)
            logging.info('Finished reading pdf file with {0}', len(pages))

            return pages

        except:
            pass

