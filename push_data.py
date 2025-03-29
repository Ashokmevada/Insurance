import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL=os.getenv("MONGODB_URL")
print(MONGO_DB_URL)
import certifi
# When you work with libraries that establish HTTPS connections (like requests), the library needs to verify that the SSL certificate presented by the server is issued by a trusted Certificate Authority.
# where searches for the absolute file path of the CA certificates used in the certifi module and assign it to the ca variable.
ca=certifi.where()
import pandas as pd
import numpy as np
import pymongo
from src.logging.logger import logging
from src.exception.exception import CustomException

class DataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json
            .loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    FILE_PATH="Data\Data.csv" # Here file is already in this path.
    DATABASE="Insurance_data"
    Collection="Data"
    networkobj=DataExtract()
    records=networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=networkobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)
