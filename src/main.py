from src.components.embedding_db import DataBaseCreation
from src.components.data_ingestion import DataIngestion

def main():
    # Creating an instance of MyClass
    data_obj = DataIngestion()
    pages = data_obj.initiate_data_ingestion()
    db_obj = DataBaseCreation(index_name='retrieval-augmentation')
    vec_db = db_obj.populating_db(pages)
    query = "Q1: What is the doses of vincristine in the high risk patients with classical Hodgkin lymphoma "
    db_obj.initiate_objects(vec_db, query)
    # print(response)
    # print(len(pages))

    # Using the class methods
    # print(obj.get_value())  # Output: 42

if __name__ == "__main__":
    main()
