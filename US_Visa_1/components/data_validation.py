import os
import sys
import pandas as pd
from pandas import DataFrame

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from US_Visa_1.exception import USvisaException
from US_Visa_1.logger import logging
from US_Visa_1.utils.main_utils import read_yaml_file, write_yaml_file
from US_Visa_1.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from US_Visa_1.entity.config_entity import DataValidationConfig
from US_Visa_1.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:  
            raise USvisaException(e,sys) from e
        


    def read_data(self,file_path: str) -> DataFrame:
       try:
           return pd.read_csv(file_path)
       except Exception as e:
              raise USvisaException(e, sys) from e
       
       
    def validate_number_of_columns(self,df:DataFrame) -> bool:
        try:
            expected_cols = len(self.schema["columns"])
            actual_cols = len(df.columns)
            status = expected_cols == actual_cols
            logging.info(f"Expected columns: {expected_cols}, Actual columns: {actual_cols}")
            return status
        except Exception as e:
            raise USvisaException(e,sys) from e
        
    def validate_column_existence(self,df:DataFrame)-> bool:
        try:
           
           missing_cols = []

           for col in self.schema['numerical_columns']:
               if col not in df.columns:
                     missing_cols.append(col)
            
           for col in self.schema['categorical_columns']:
                if col not in df.columns:
                     missing_cols.append(col)

           if missing_cols: # alternatively we can write if len(missing_cols) > 0:
                logging.info(f"Missing columns: {missing_cols}")
                return False
           
           return True
        except Exception as e:
            raise USvisaException(e,sys) from e
        

    def detect_data_drift(self,reference_df:DataFrame,current_df=DataFrame) -> bool:
        """
        Detect dataset drift using Evidently
        Saves HTML + YAML report
        Returns True if drift detected else False
        """
        try:
            logging.info("Initiating data drift detection,Running Evidently data drift check")

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df,current_data=current_df)

            # ✅ Ensure drift report directory exists
            os.makedirs(
                self.data_validation_config.drift_report_dir,
                exist_ok=True
            )

            # Save HTML report
            report.save_html(
                self.data_validation_config.drift_report_html_path
            )

            # Save YAML / JSON report
            report_json = report.as_dict()
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report_json,
            )

            drift_status = report_json["metrics"][0]["result"]["dataset_drift"]
            drift_share = report_json["metrics"][0]["result"]["drift_share"]

            logging.info(f"Drift detected: {drift_status}")
            logging.info(f"Drift share: {drift_share}")

            return drift_status

        except Exception as e:
            raise USvisaException(e, sys)
        

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df =self.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info("Read train and test data for validation")
            # Validate number of columns

            error_msg =""

            if not self.validate_number_of_columns(train_df):
                error_msg += "Train data does not contain all the required columns.\n"
            if not self.validate_number_of_columns(test_df):
                error_msg += "Test data does not contain all the required columns.\n"
            # Validate column existence
            if not self.validate_column_existence(train_df):
                error_msg += "Train data does not contain all the required columns.\n"

            if not self.validate_column_existence(test_df):
                error_msg += "Test data does not contain all the required columns.\n"

            validation_status = len(error_msg) == 0

            if validation_status:
                drift_detected = self.detect_data_drift(train_df,test_df)
                if drift_detected:
                    logging.info("Data drift detected between train and test datasets")
                else:
                    logging.info("No data drift detected between train and test datasets")

            data_validation_artifact = DataValidationArtifact(validation_status=validation_status,
                                                              message=error_msg,
                                                              drift_report_file_path=self.data_validation_config.drift_report_file_path,
                drift_report_html_path=self.data_validation_config.drift_report_html_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e,sys) from e
            