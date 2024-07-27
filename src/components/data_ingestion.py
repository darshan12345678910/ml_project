import os
import sys
sys.path.append('E:\ml_project')
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_conf=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("entered Data Ingestion method or component")
        try:
            df=pd.read_csv("notebook\data\stud.csv")
            logging.info("Read Data sucessfuly as Dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_conf.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_conf.raw_data_path,index=False,header=True)
            logging.info("Intiated the train,test split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion_conf.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_conf.test_data_path,index=False,header=True)
            logging.info("the data ingestion is completed")
            return(
                self.data_ingestion_conf.train_data_path,
                self.data_ingestion_conf.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

         
