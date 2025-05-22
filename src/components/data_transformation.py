import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.logging.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import CustomException 
from src.utils.main_utils.utils import save_numpy_array_data, save_object
def label_encoder_function(df):
            for col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col])
            return df

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    
    
    def get_data_transformer_object(self) -> ColumnTransformer:

        try:
            ordinal_cols = ["Prior_Insurance", "Claims_Severity"]
            label_cols = ["Policy_Type", "Source_of_Lead", "Region", "Marital_Status"]
            numerical_cols = ['Age', 'Married_Premium_Discount', 'Claims_Adjustment',
                            'Premium_Amount', 'Safe_Driver_Discount',
                            'Multi_Policy_Discount', 'Bundling_Discount',
                            'Total_Discounts', 'Credit_Score']

            # Define order for Ordinal Encoding
            prior_insurance_order = ['<1 year', '1-5 years', '>5 years']
            claims_severity_order = ['Low', 'Medium', 'High']

            ordinal_pipeline = Pipeline(steps=[
                ('ordinal_encoder', OrdinalEncoder(categories=[prior_insurance_order, claims_severity_order])),
                ('scaler', StandardScaler())
            ])

            label_pipeline = Pipeline(steps=[
                ('label_encoder', FunctionTransformer(label_encoder_function, validate=False)),
                ('scaler', StandardScaler())
            ])

            numerical_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('ordinal', ordinal_pipeline, ordinal_cols),
                ('label', label_pipeline, label_cols),
                ('numerical', numerical_pipeline, numerical_cols)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Started Data Transformation and Reading train and test file")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            ordinal_cols = ["Prior_Insurance", "Claims_Severity"]
            label_cols = ["Policy_Type", "Source_of_Lead", "Region", "Marital_Status"]
            drop_columns = ["Is_Senior", "Policy_Adjustment", "Prior_Insurance_Premium_Adjustment", "Time_Since_First_Contact", 
                            "Conversion_Status", "Premium_Adjustment_Region", "Premium_Adjustment_Credit", "Time_to_Conversion", 
                            "Quotes_Requested", "Inquiries", "Website_Visits"]
            numerical_cols = ['Age', 'Married_Premium_Discount', 'Claims_Adjustment',
                            'Premium_Amount', 'Safe_Driver_Discount',
                            'Multi_Policy_Discount', 'Bundling_Discount',
                            'Total_Discounts', 'Credit_Score']
            
            preprocessor = self.get_data_transformer_object()

            train_df.drop(columns=drop_columns, inplace=True)
            test_df.drop(columns=drop_columns, inplace=True)

            transformed_train = preprocessor.fit_transform(train_df)
            transformed_test = preprocessor.transform(test_df)

            # Convert to numpy arrays after transformations
            train_arr = transformed_train
            test_arr = transformed_test
                        

            transformed_train_df = pd.DataFrame(transformed_train, columns=ordinal_cols + label_cols + numerical_cols )
            transformed_test_df = pd.DataFrame(transformed_test, columns=ordinal_cols + label_cols + numerical_cols)
            
            
            for col in ordinal_cols + label_cols:

                train_df[col] = transformed_train_df[col]
                test_df[col] = transformed_test_df[col]
            
            # Reorder columns to place 'Claims_Frequency' as the last column
            if "Claims_Frequency" in train_df.columns:
                train_df = train_df[[col for col in train_df.columns if col != "Claims_Frequency"] + ["Claims_Frequency"]]
            if "Claims_Frequency" in test_df.columns:
                test_df = test_df[[col for col in test_df.columns if col != "Claims_Frequency"] + ["Claims_Frequency"]]

            train_arr = train_df.to_numpy()
            test_arr = test_df.to_numpy()
            
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object("final_model/preprocessor.pkl", preprocessor)
            
            
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':

    try: 

        logging.info("Starting Data Transformation Stage")
        from src.entity.config_entity import TrainingpipelineConfig, DataValidationConfig, DataTransformationConfig, DataIngestionConfig
        from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact

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
        
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        logging.info("Completed Data Transformation Stage")


    except Exception as e:
        raise CustomException(e, sys)
