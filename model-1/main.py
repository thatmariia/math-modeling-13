from Constants import *
from DataConstructor import DataConstructor
from TrainerAgent import TrainerAgent

if __name__ == '__main__':

    # constructing label + pixel dataframes for training and tests
    dataConstructor = DataConstructor()
    dataConstructor.construct()

    trainerAgent = TrainerAgent(train=dataConstructor.df, test=dataConstructor.df_test)
    trainerAgent.perform()