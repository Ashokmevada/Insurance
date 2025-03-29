from src.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact
from src import constants
from src.entity.config_entity import DataValidationConfig, DataIngestionConfig
from src.exception.exception import CustomException
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file
from src.logging.logger import logging
from scipy.stats import ks_2samp
import pandas as pd
import os,sys


class DataValidation:

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact , data_validation_config: DataValidationConfig):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(constants.SCHEMA_FILE_PATH)
            

        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
    
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        """
        Validate number of columns in the dataframe
        
        """

        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomException(e,sys)
        
    def detect_dataset_drift(self,base_df,current_df, threshold=0.05)->bool:

        """
        Detects data drift in the new data by comparing distribution of the data
        
        """

        try:
            status = True
            report ={}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dis = ks_2samp(d1,d2)
                if threshold <= is_same_dis.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update( { column: {
                    "p_value" : float(is_same_dis.pvalue),
                    "drift_status" : is_found
                }})

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path , exist_ok=True)
            write_yaml_file(file_path = drift_report_file_path, content = report)

            return status
                             

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_validation(self) -> DataValidationArtifact:

        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            #Read the data from train and test
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            #validate number of columns

            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all columns.\n"

            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all columns.\n"

            # lets check datadrift
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            if not status:
                error_message = f"Data drift found between train and test dataframe.\n"
                raise Exception(error_message)

            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_test_file_path=None,
                invalid_train_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        
        except Exception as e:
            raise CustomException(e,sys)




        

