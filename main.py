from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.entity.config_entity import TrainingpipelineConfig, DataIngestionConfig , DataValidationConfig, DataTransformationConfig,ModelTrainerConfig
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

        logging.info("Model Training started...")
        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")

    except Exception as e:
        raise CustomException(e, sys)