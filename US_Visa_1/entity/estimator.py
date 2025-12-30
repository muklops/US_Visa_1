import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from US_Visa_1.exception import USvisaException
from US_Visa_1.logger import logging




class TargetValueMapping: # To map target y vales ['Certified', 'Denied'] to [0,1],since computer know only numbers
    def __init__(self):
        self.Certified:int = 0
        self.Denied:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))