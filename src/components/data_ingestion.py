from src.exception import exception
from src.logging.logger import logging
## Data Ingestion Configs
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import shutil
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from src.entity.config_entity import TrainingpipelineConfig
load_dotenv()
MONGO_DB_URL=os.getenv("MONGODB_URL")
class DataIngestion:    
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise exception.CustomException(e, sys)
        
    def export_collection_as_dataframe(self):
        """
        Read data from Mongodb and Return a dataframe by removing the _id column and replacing na with np.nan
        
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            # whenever we fetch data from mongodb, we get _id column. so we need to drop it
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"] , axis=1)
            
            df.replace({"na", np.nan} , inplace = True)
            return df
            
        
        except Exception as e:
            raise exception.CustomException(e, sys)
        
    
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            # ✅ Also save to fixed path (for DVC)
            latest_raw_path = os.path.join("data", "raw", "latest")
            if os.path.exists(latest_raw_path):
                shutil.rmtree(latest_raw_path)
            shutil.copytree(dir_path, latest_raw_path)

            logging.info(f"Raw data saved to {dir_path} and synced to latest/")
            return dataframe

            return dataframe
        except Exception as e:
            raise exception.CustomException(e, sys)
        
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set , test_set = train_test_split(dataframe , test_size = self.data_ingestion_config.train_test_split_ratio)
            logging.info(f"Performed Train Test Split on the dataframe")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path , index = False, header = True)
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index = False,
                header = True
            )
            logging.info(f"Exported train and test file path.")
        
        except Exception as e:
            
            raise exception.CustomException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifacts = DataIngestionArtifact(
                trained_file_path = self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path
            )
            return dataingestionartifacts

        except Exception as e:
            raise exception.CustomException(e, sys)


if __name__ == "__main__":

    try:

        training_pipeline_config = TrainingpipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiated Data Ingestion")
        data_ingestion_artifacts =data_ingestion.initiate_data_ingestion()

    except Exception as e:
        raise exception.CustomException(e, sys)