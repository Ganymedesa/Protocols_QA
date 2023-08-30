from src.components.embedding_db import DataBaseCreation
from src.components.data_ingestion import DataIngestion

def main():
    # Creating an instance of MyClass
    data_obj = DataIngestion()
    pages = data_obj.initiate_data_ingestion()

    db_obj = DataBaseCreation('protocol-retrieval-augmentation')
    db_obj.initialize_vec_db()
    vec_db = db_obj.populating_db(pages)

    query = "Question: High risk patients in CMR end of 2 cycles but he had initial large mediastinal lymphadenopathy should he receive radiation?"
    ans = db_obj.initiate_objects(vec_db, query)
    print(ans)
    # res = vec_db.similarity_search(
    #     query,  # our search query
    #     k=3  # return 3 most relevant docs
    #     )
    # print(res)
    

if __name__ == "__main__":
    main()
