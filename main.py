from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import TrainingpipelineConfig, DataIngestionConfig , DataValidationConfig, DataTransformationConfig
from src.logging.logger import logging
from src.exception.exception import CustomException
import sys

if __name__ == "__main__":

    try:

        training_pipeline_config = TrainingpipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiated Data Ingestion")
        data_ingestion_artifacts =data_ingestion.initiate_data_ingestion()
        logging.info("Completed Data Ingestion")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifacts,data_validation_config)
        logging.info("Initiated Data Validation")
        data_validation_artifacts = data_validation.initiate_data_validation()
        logging.info("Completed Data Validation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        logging.info("Initiated Data Transformation")
        data_transformation = DataTransformation(data_validation_artifacts, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Completed Data Transformation")


    except Exception as e:
        raise CustomException(e, sys)