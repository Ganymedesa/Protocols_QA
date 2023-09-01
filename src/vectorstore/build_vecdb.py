from src.vectorstore.embedding_db import DataBaseCreation
from src.vectorstore.data_ingestion import DataIngestion

def build_vecdb():
    # Creating an instance of MyClass
    data_obj = DataIngestion()
    pages = data_obj.initiate_data_ingestion()

    db_obj = DataBaseCreation('protocol-retrieval-augmentation')
    db_obj.initialize_vec_db()
    vec_db = db_obj.populating_db(pages)

    
build_vecdb()
