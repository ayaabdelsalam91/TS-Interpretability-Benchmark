import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import Helper
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from Helper import  checkAccuracy
import os
import time
import random

from Plotting import *


maskedPercentages=[ i for i in range(0,101,10)]



def main(args,DatasetsTypes,DataGenerationTypes,models,device):
    if  os.path.exists(args.ignore_list):
        f = open(args.ignore_list, 'r+')
        ignore_list = [line for line in f.readlines()]
        f.close()
        for i in range(len(ignore_list)):
            if('\n' in ignore_list[i]):
                ignore_list[i]=ignore_list[i][:-1]
    else:
        ignore_list=[]



    Saliency_Methods=[]

    if(args.GradFlag):
        Saliency_Methods.append("Grad")
    if(args.IGFlag):
        Saliency_Methods.append("IG")
    if(args.DLFlag):
        Saliency_Methods.append("DL")
    if(args.GSFlag):
        Saliency_Methods.append("GS")
    if(args.DLSFlag):
        Saliency_Methods.append("DLS")
    if(args.SGFlag):
        Saliency_Methods.append("SG")
    if(args.ShapleySamplingFlag):
        Saliency_Methods.append("ShapleySampling")
    if(args.FeaturePermutationFlag):
        Saliency_Methods.append("FeaturePermutation")
    if(args.FeatureAblationFlag):
        Saliency_Methods.append("FeatureAblation")
    if(args.OcclusionFlag):
        Saliency_Methods.append("Occlusion")

    Saliency_Methods.append("Random")

    for x in range(len(DatasetsTypes)):
        for y in range(len(DataGenerationTypes)):
            args.DataGenerationProcess=DataGenerationTypes[y]
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
            raw_Testing=np.copy(Testing)

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

            # save np.load
            np_load_old = np.load

            # modify the default parameters of np.load
            np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


            modelName="Simulated"
            modelName+=args.DataName

            for m in range(len(models)):
                start = time.time()
                resultFileName=args.Masked_Acc_dir + args.DataName+"_"+models[m]


                Y_DimOfGrid=len(maskedPercentages)+1
                X_DimOfGrid=len(Saliency_Methods)

                Grid = np.zeros((X_DimOfGrid,Y_DimOfGrid),dtype='object')

                Grid[:,0]=Saliency_Methods
                columns=["saliency method"]
                for mask in maskedPercentages:
                    columns.append(str(mask))


                if(args.DataName+"_"+models[m] in ignore_list):
                    print("ignoring",args.DataName+"_"+models[m]  )
                    continue
                
                else:

                    saveModelName="../Models/"+models[m]+"/"+modelName
                    saveModelBestName =saveModelName +"_BEST.pkl"



                    pretrained_model = torch.load(saveModelBestName,map_location=device) 
                    Test_Unmasked_Acc  =   checkAccuracy(test_loaderRNN , pretrained_model, args)


                    for s,saliency in enumerate(Saliency_Methods):
                        Test_Masked_Acc=Test_Unmasked_Acc
                        for i , maskedPercentage in enumerate(maskedPercentages):
                            
                            start_percentage=time.time()
                            if(maskedPercentage==0):
                                Grid[s][i+1]=Test_Unmasked_Acc
                            elif(Test_Masked_Acc==0):
                                Grid[s][i+1]=Test_Masked_Acc
                            else:
                                if(maskedPercentage !=100):
                                    mask = np.load(args.Mask_dir+modelName+"_"+models[m]+"_"+saliency+"_"+str(maskedPercentage)+"_percentSal_rescaled.npy")

                                    toMask=np.copy(raw_Testing)
                                    MaskedTesting=Helper.maskData(args,toMask,mask,True)
                                    MaskedTesting=scaler.transform(MaskedTesting)
                                    MaskedTesting=MaskedTesting.reshape(-1,args.NumTimeSteps,args.NumFeatures)

                                else:

                                    MaskedTesting=np.zeros((Testing.shape[0] , args.NumTimeSteps*args.NumFeatures))
                                    sample = Helper.generateNewSample(args).reshape(args.NumTimeSteps*args.NumFeatures)
                                    MaskedTesting[:,:]= sample

                                    MaskedTesting=scaler.transform(MaskedTesting)
                                    MaskedTesting=MaskedTesting.reshape(-1,args.NumTimeSteps,args.NumFeatures)

                                if(args.plot):

                                    randomIndex = 10
                                    plotExampleBox(MaskedTesting[randomIndex],args.Graph_dir+args.DataName+"_"+models[m]+"_"+saliency+"_percentMasked"+str(maskedPercentage),flip=True)


                                Maskedtest_dataRNN = data_utils.TensorDataset(torch.from_numpy(MaskedTesting),torch.from_numpy( TestingLabel))
                                Maskedtest_loaderRNN = data_utils.DataLoader(Maskedtest_dataRNN, batch_size=args.batch_size, shuffle=False)

                                Test_Masked_Acc  =   checkAccuracy(Maskedtest_loaderRNN , pretrained_model,args)
                                print('{} {} {} Acc {:.2f} Masked Acc {:.2f} Highest Value mask {}'.format(args.DataName,models[m],saliency,Test_Unmasked_Acc ,Test_Masked_Acc,maskedPercentage))
                                Grid[s][i+1]=Test_Masked_Acc
                            end_percentage=time.time()
                    end = time.time()
                    print('{} {} time: {}'.format(args.DataName,models[m],end-start))
                    print()



                    for percent in maskedPercentages:

                        resultFileName=resultFileName+"_"+str(percent)
                    resultFileName=resultFileName+"_percentSal_rescaled.csv"
                    Helper.save_intoCSV(Grid,resultFileName,col=columns)




            np.load = np_load_old
