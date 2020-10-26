import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import Helper
from Helper import checkAccuracy  
import random



import logging

import warnings
warnings.filterwarnings("ignore")





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
              



