from src.components.embedding_db import DataBaseCreation
from src.components.data_ingestion import DataIngestion

def main():
    # Creating an instance of MyClass
    obj = DataIngestion()
    pages = obj.initiate_data_ingestion()
    print(len(pages))

    # Using the class methods
    # print(obj.get_value())  # Output: 42

if __name__ == "__main__":
    main()
