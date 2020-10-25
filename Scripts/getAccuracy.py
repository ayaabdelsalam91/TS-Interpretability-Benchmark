import torch
import argparse
import sys

sys.path.append('./Models/')
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import Helper
from Helper import checkAccuracy  
import random
import time
from sklearn import preprocessing
from  sklearn.preprocessing import minmax_scale

from Transformer import Transformer
from LSTMWithInputCellAttention import LSTMWithInputCellAttention
from LSTM import LSTM
from TCN import TCN
from Plotting import *


import logging

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






DatasetsTypes= ["Middle", "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]
DataGenerationTypes=[None,"Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ]


DatasetsTypes= ["SmallMiddle"]
DataGenerationTypes=["Harmonic"]
# models=["LSTM","LSTMWithInputCellAttention","TCN","Transformer"]


models=["LSTM","LSTMWithInputCellAttention","TCN","Transformer"]


def main(args):
    for m in range(len(models)):

        for x in range(len(DatasetsTypes)):
            for y in range(len(DataGenerationTypes)):

                if(DataGenerationTypes[y]==None):
                    args.DataName=DatasetsTypes[x]+"_Box"
                else:
                    args.DataName=DatasetsTypes[x]+"_"+DataGenerationTypes[y]

                Training=np.load(args.data_dir+"SimulatedTrainingData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
                TrainingMetaDataset=np.load(args.data_dir+"SimulatedTrainingMetaData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
                TrainingLabel=TrainingMetaDataset[:,0]

                Testing=np.load(args.data_dir+"SimulatedTestingData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
                TestingDataset_MetaData=np.load(args.data_dir+"SimulatedTestingMetaData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
                TestingLabel=TestingDataset_MetaData[:,0]



                Training = Training.reshape(Training.shape[0],Training.shape[1]*Training.shape[2])
                Testing = Testing.reshape(Testing.shape[0],Testing.shape[1]*Testing.shape[2])


                scaler = MinMaxScaler()
                scaler.fit(Training)
                Training = scaler.transform(Training)
                Testing = scaler.transform(Testing)

                TrainingRNN = Training.reshape(Training.shape[0] , args.NumTimeSteps,args.NumFeatures)
                TestingRNN = Testing.reshape(Testing.shape[0] , args.NumTimeSteps,args.NumFeatures)



                train_dataRNN = data_utils.TensorDataset(torch.from_numpy(TrainingRNN), torch.from_numpy(TrainingLabel))
                train_loaderRNN = data_utils.DataLoader(train_dataRNN, batch_size=args.batch_size, shuffle=True)


                test_dataRNN = data_utils.TensorDataset(torch.from_numpy(TestingRNN),torch.from_numpy( TestingLabel))
                test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=args.batch_size, shuffle=False)
    


                modelName="Simulated"
                modelName+=args.DataName



                saveModelName="../Models/"+models[m]+"/"+modelName
                saveModelBestName =saveModelName +"_BEST.pkl"



                pretrained_model = torch.load(saveModelBestName,map_location=device) 
                Test_Acc  =   checkAccuracy(test_loaderRNN , pretrained_model, args)
                print('{} {} model BestAcc {:.4f}'.format(args.DataName,models[m],Test_Acc))
                if(Test_Acc<90):
                    logging.basicConfig(filename=args.log_file,level=logging.DEBUG)

                    logging.debug('{} {} model BestAcc {:.4f}'.format(args.DataName,models[m],Test_Acc))







def parse_arguments(argv):


    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default="../Datasets/")
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--logging', type=bool, default=True)


    parser.add_argument('--Graph_dir', type=str, default='../Graphs/')
    parser.add_argument('--Results_dir', type=str, default='../Results/Saliency_Values/')
    parser.add_argument('--log_file', type=str, default='acc_under_90.log')



    parser.add_argument('--NumTimeSteps',type=int,default=50)
    parser.add_argument('--NumFeatures',type=int,default=50)
    parser.add_argument('--d_a', type=int, default=50)
    parser.add_argument('--attention_hops', type=int, default=10)

    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=5)


    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--levels', type=int,default=3)


    parser.add_argument('--hidden_size', type=int,default=5)
    parser.add_argument('--batch_size', type=int,default=10)
    parser.add_argument('--num_epochs', type=int,default=400)
    parser.add_argument('--learning_rate', type=float,default=0.001)
    parser.add_argument('--rnndropout', type=float,default=0.1)



    parser.add_argument('--GradFlag', type=bool, default=True)
    parser.add_argument('--IGFlag', type=bool, default=True)
    parser.add_argument('--DLFlag', type=bool, default=True)
    parser.add_argument('--GSFlag', type=bool, default=True)
    parser.add_argument('--DLSFlag', type=bool, default=True)
    parser.add_argument('--SGFlag', type=bool, default=True)
    parser.add_argument('--ShapleySamplingFlag', type=bool, default=True)
    parser.add_argument('--FeaturePermutationFlag', type=bool, default=True)
    parser.add_argument('--FeatureAblationFlag', type=bool, default=True)
    parser.add_argument('--OcclusionFlag', type=bool, default=True)






    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))