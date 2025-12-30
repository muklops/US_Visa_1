import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
)
from sklearn.compose import ColumnTransformer

from US_Visa_1.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from US_Visa_1.entity.config_entity import DataTransformationConfig
from US_Visa_1.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from US_Visa_1.exception import USvisaException
from US_Visa_1.logger import logging
from US_Visa_1.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
    drop_columns,
)
from US_Visa_1.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore")
            ordinal_encoder = OrdinalEncoder()

            transform_pipe = Pipeline(
                steps=[("transformer", PowerTransformer(method="yeo-johnson"))]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("onehot", oh_transformer, self._schema_config["oh_columns"]),
                    ("ordinal", ordinal_encoder, self._schema_config["or_columns"]),
                    ("power", transform_pipe, self._schema_config["transform_columns"]),
                    ("scaler", numeric_transformer, self._schema_config["num_features"]),
                ]
            )
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation")

            preprocessor = self.get_data_transformer_object()

            # -------------------------
            # Load data
            # -------------------------
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # -------------------------
            # Split features & target
            # -------------------------
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # -------------------------
            # Feature engineering
            # -------------------------
            X_train["company_age"] = CURRENT_YEAR - X_train["yr_of_estab"]
            X_test["company_age"] = CURRENT_YEAR - X_test["yr_of_estab"]

            drop_cols = self._schema_config["drop_columns"]
            X_train = drop_columns(X_train, drop_cols)
            X_test = drop_columns(X_test, drop_cols)

            y_train = y_train.replace(TargetValueMapping()._asdict())
            y_test = y_test.replace(TargetValueMapping()._asdict())

            # -------------------------
            # Preprocessing
            # -------------------------
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # -------------------------
            #  SMOTEENN → TRAIN ONLY
            # -------------------------
            logging.info("Applying SMOTEENN on TRAINING data only")

            smt = SMOTEENN(sampling_strategy="minority")
            X_train_final, y_train_final = smt.fit_resample(
                X_train_arr, y_train
            )

            #  TEST DATA NOT TOUCHED(SMOTEENN applied only in Train sets)
            X_test_final = X_test_arr
            y_test_final = y_test

            # -------------------------
            # Combine X & y
            # -------------------------
            train_arr = np.c_[X_train_final, y_train_final]
            test_arr = np.c_[X_test_final, y_test_final]

            # -------------------------
            # Save artifacts
            # -------------------------
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr,
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr,
            )

            logging.info("Data transformation completed successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as e:
            raise USvisaException(e, sys)
