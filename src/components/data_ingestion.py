import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# any specific input is required
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") # output files will be stored in artifacts folder
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()# input we need to initialize
        # on calling DataIngestion class, these 3 paths will get saved inside the ingestion_config class variable

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # read data
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info("Read dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            # convert raw data to csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            #split into train and test
            logging.info("Train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("Data ingestion completed")

            return ( # return data for transformation phase
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation() # it will call init and create the config file
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))