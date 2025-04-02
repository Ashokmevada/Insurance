import os
import sys
from src.constants import SAVED_MODEL_DIR,MODEL_FILE_NAME
from src.exception.exception import CustomException
from src.logging.logger import logging

class Model:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CustomException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise CustomException(e,sys)