import os
import sys

from src.exception.exception import CustomException 
from src.logging.logger import logging

from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig



from src.utils.ml_utils.model.estimator import Model
from src.utils.main_utils.utils import save_object,load_object
from src.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from src.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse
import numpy as np
import pandas as pd

import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.init(repo_owner='ashokmevada18', repo_name='Insurance', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"]= os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"]= os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"]= os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def track_mlflow(self, best_model, classificationmetric, best_model_params=None):
        mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Only start a single run to log everything under one experiment
        with mlflow.start_run():

            # Log metrics
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)

            # Log the model
            mlflow.sklearn.log_model(best_model, "model")

            # Log hyperparameters if provided
            if best_model_params:
                mlflow.log_params(best_model_params)

            # Register the model if necessary
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name="best_model")
            else:
                mlflow.sklearn.log_model(best_model, "model")



        
    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1, n_jobs=-1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, verbose=1),
            "AdaBoost": AdaBoostClassifier()
        }

        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 10, 20, None]
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None]
            },
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05],
                'n_estimators': [50, 100, 200],
                'subsample': [0.6, 0.75, 0.9]
            },
            "Logistic Regression": {
                'C': [0.01, 0.1, 1, 10]
            },
            "AdaBoost": {
                'learning_rate': [0.01, 0.1, 1],
                'n_estimators': [50, 100, 200]
            }
        }

        logging.info("Starting model evaluation...")
        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params
        )

        best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
        best_model_score = model_report[best_model_name]["score"]
        best_model = models[best_model_name]
        best_model_params = model_report[best_model_name]["best_params"]

        logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
        logging.info(f"Best hyperparameters: {best_model_params}")

        # Train final model with best params
        best_model.set_params(**best_model_params)
        best_model.fit(X_train, y_train)

        y_train_pred = best_model.predict(X_train)
        train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(X_test)
        test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # Log using MLflow
        self.track_mlflow(best_model, train_metric, best_model_params)
        self.track_mlflow(best_model, test_metric, best_model_params)

        # Save final model with preprocessor
        preprocessor = load_object("final_model/preprocessor.pkl")
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        model = Model(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=model)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

        
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
            try:
                train_file_path = self.data_transformation_artifact.transformed_train_file_path
                test_file_path = self.data_transformation_artifact.transformed_test_file_path

                #loading training array and testing array
                train_arr = load_numpy_array_data(train_file_path)
                test_arr = load_numpy_array_data(test_file_path)

                x_train, y_train, x_test, y_test = (
                    train_arr[:, :-1],
                    train_arr[:, -1],
                    test_arr[:, :-1],
                    test_arr[:, -1],
                )

                model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
                return model_trainer_artifact

                
            except Exception as e:
                raise CustomException(e,sys)

if __name__ == '__main__':

    try: 

        logging.info("Starting Model Training Stage")
        from src.entity.config_entity import TrainingpipelineConfig, DataValidationConfig, DataTransformationConfig, DataIngestionConfig
        from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact, DataTransformationArtifact

        training_pipeline_config = TrainingpipelineConfig()
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_validation_config = DataValidationConfig(training_pipeline_config)

        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)

        data_ingestion_artifacts = DataIngestionArtifact(
                trained_file_path = data_ingestion_config.training_file_path,
                test_file_path = data_ingestion_config.testing_file_path
            )

        data_validation_artifact = DataValidationArtifact(
        validation_status=True,
        valid_train_file_path=data_ingestion_artifacts.trained_file_path,
        valid_test_file_path=data_ingestion_artifacts.test_file_path,
        invalid_train_file_path="",  # if not needed, can be dummy or empty
        invalid_test_file_path="",   # same here
        drift_report_file_path=""    # only needed if used in transformation
    )
        
        data_transformation_artifact = DataTransformationArtifact(
            transformed_object_file_path=data_transformation_config.transformed_object_file_path,
            transformed_train_file_path=data_transformation_config.transformed_train_file_path,
            transformed_test_file_path=data_transformation_config.transformed_test_file_path
        )

        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        
       
        logging.info("Completed Model Training Stage")


    except Exception as e:
        raise CustomException(e, sys)
