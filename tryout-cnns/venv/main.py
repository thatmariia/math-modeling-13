from DataConstructor import DataConstructor
from TrainerAgent import TrainerAgent
from Constants import *



if __name__ == '__main__':

    # constructing label + pixel dataframes for training and tests
    dataConstructor = DataConstructor()
    dataConstructor.construct()
    df_train = dataConstructor.df_train
    df_test  = dataConstructor.df_test

    trainerAgent = TrainerAgent(train=df_train, test=df_test)
    trainerAgent.perform()
