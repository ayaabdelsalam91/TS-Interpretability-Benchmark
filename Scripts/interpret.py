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
import os
import logging

import warnings
warnings.filterwarnings("ignore")

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    Saliency,
    NoiseTunnel,
    ShapleyValueSampling,
    FeaturePermutation,
    FeatureAblation,
    Occlusion

)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# DatasetsTypes= ["Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]



DatasetsTypes= ["Middle", "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]
DataGenerationTypes=[None,"Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ]



models=["Transformer","LSTMWithInputCellAttention","TCN","Transformer"]


models=["LSTM"]
# ,"LSTMWithInputCellAttention","TCN","Transformer"]


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

            
                if(Test_Acc>=90):

                    if(args.GradFlag):
                        rescaledGrad= np.zeros((TestingRNN.shape))
                        Grad = Saliency(pretrained_model)

                    if(args.IGFlag):
                        rescaledIG= np.zeros((TestingRNN.shape))
                        IG = IntegratedGradients(pretrained_model)
                    if(args.DLFlag):
                        rescaledDL= np.zeros((TestingRNN.shape))
                        DL = DeepLift(pretrained_model)
                    if(args.GSFlag):
                        rescaledGS= np.zeros((TestingRNN.shape))
                        GS = GradientShap(pretrained_model)
                    if(args.DLSFlag):
                        rescaledDLS= np.zeros((TestingRNN.shape))
                        DLS = DeepLiftShap(pretrained_model)
                                
                    if(args.SGFlag):
                        rescaledSG= np.zeros((TestingRNN.shape))
                        Grad_ = Saliency(pretrained_model)
                        SG = NoiseTunnel(Grad_)


                    if(args.ShapleySamplingFlag):
                        rescaledShapleySampling= np.zeros((TestingRNN.shape))
                        SS = ShapleyValueSampling(pretrained_model)
                    if(args.GSFlag):
                        rescaledFeaturePermutation= np.zeros((TestingRNN.shape))
                        FP = FeaturePermutation(pretrained_model)
                    if(args.FeatureAblationFlag):
                        rescaledFeatureAblation= np.zeros((TestingRNN.shape))
                        FA = FeatureAblation(pretrained_model)
                                
                    if(args.OcclusionFlag):
                        rescaledOcclusion= np.zeros((TestingRNN.shape))
                        OS = Occlusion(pretrained_model)


                    idx=0
                    mask=np.zeros((args.NumTimeSteps, args.NumFeatures),dtype=int)
                    for i in  range (args.NumTimeSteps):
                        mask[i,:]=i

                    for i,  (samples, labels)  in enumerate(test_loaderRNN):


                        if(i%2==0):
                            print('[{}/{}] {} {} model accuracy {:.2f}'\
                                .format(i,len(test_loaderRNN), models[m], args.DataName, Test_Acc))


                            input = samples.reshape(-1, args.NumTimeSteps, args.NumFeatures).to(device)
                            input = Variable(input,  volatile=False, requires_grad=True)

                            batch_size = input.shape[0]
                            baseline_single=torch.from_numpy(np.random.random(input.shape)).to(device)
                            baseline_multiple=torch.from_numpy(np.random.random((input.shape[0]*5,input.shape[1],input.shape[2]))).to(device)
                            inputMask= np.zeros((input.shape))
                            inputMask[:,:,:]=mask
                            inputMask =torch.from_numpy(inputMask).to(device)
                            mask_single= torch.from_numpy(mask).to(device)
                            mask_single=mask_single.reshape(1,args.NumTimeSteps, args.NumFeatures).to(device)
                            labels=torch.tensor(labels.int().tolist()).to(device)





                            if(args.GradFlag):
                                attributions = Grad.attribute(input, \
                                                              target=labels)
                                rescaledGrad[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions)


                            if(args.IGFlag):
                                attributions = IG.attribute(input,  \
                                                            baselines=baseline_single, \
                                                            target=labels)
                                rescaledIG[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions)




                            if(args.DLFlag):
                                attributions = DL.attribute(input,  \
                                                            baselines=baseline_single, \
                                                            target=labels)
                                rescaledDL[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions)





                            if(args.GSFlag):

                                attributions = GS.attribute(input,  \
                                                            baselines=baseline_multiple, \
                                                            stdevs=0.09,\
                                                            target=labels)
                                rescaledGS[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions)


                            if(args.DLSFlag):

                                attributions = DLS.attribute(input,  \
                                                            baselines=baseline_multiple, \
                                                            target=labels)
                                rescaledDLS[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions)




                            if(args.SGFlag):
                                attributions = SG.attribute(input, \
                                                            target=labels)
                                rescaledSG[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions)



                            if(args.ShapleySamplingFlag):
                                attributions = SS.attribute(input, \
                                                baselines=baseline_single, \
                                                target=labels,\
                                                feature_mask=inputMask)
                                rescaledShapleySampling[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions)


                            if(args.FeaturePermutationFlag):
                                attributions = FP.attribute(input, \
                                                target=labels,
                                                perturbations_per_eval= input.shape[0],\
                                                feature_mask=mask_single)
                                rescaledFeaturePermutation[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions) 


                            if(args.FeatureAblationFlag):
                                attributions = FA.attribute(input, \
                                                target=labels)
                                                # perturbations_per_eval= input.shape[0],\
                                                # feature_mask=mask_single)
                                rescaledFeatureAblation[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions) 


                            if(args.OcclusionFlag):
                                attributions = OS.attribute(input, \
                                                sliding_window_shapes=(1,args.NumFeatures),
                                                target=labels,
                                                baselines=baseline_single)
                                rescaledOcclusion[idx:idx+batch_size,:,:]=Helper.givenAttGetRescaledSaliency(args,attributions) 


                            idx+=batch_size



                    if(args.plot):
                        index = random.randint(0,TestingRNN.shape[0]-1)
                        index=30
                        plotExampleBox(TestingRNN[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_sample',flip=True)

                        print("Plotting sample",index)
                        if(args.GradFlag):
                            print("Grad",rescaledGrad[index,:,:].sum())
                            plotExampleBox(rescaledGrad[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_Grad',greyScale=True,flip=True)

                        if(args.IGFlag):
                            plotExampleBox(rescaledIG[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_IG',greyScale=True, flip=True)


                        if(args.DLFlag):
                            plotExampleBox(rescaledDL[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_DL',greyScale=True, flip=True)


                        if(args.GSFlag):
                            plotExampleBox(rescaledGS[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_GS',greyScale=True, flip=True)


                        if(args.DLSFlag):
                            plotExampleBox(rescaledDLS[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_DLS',greyScale=True, flip=True)


                        if(args.SGFlag):
                            plotExampleBox(rescaledSG[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_SG',greyScale=True, flip=True)


                        if(args.ShapleySamplingFlag):
                            plotExampleBox(rescaledShapleySampling[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_ShapleySampling',greyScale=True, flip=True)


                        if(args.FeaturePermutationFlag):
                            plotExampleBox(rescaledFeaturePermutation[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_FeaturePermutation',greyScale=True, flip=True)


                        if(args.FeatureAblationFlag):
                            plotExampleBox(rescaledFeatureAblation[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_FeatureAblation',greyScale=True, flip=True)

                        if(args.OcclusionFlag):
                            plotExampleBox(rescaledOcclusion[index,:,:],args.Graph_dir+args.DataName+"_"+models[m]+'_Occlusion',greyScale=True, flip=True)

                    if(args.save):
                        if(args.GradFlag):
                            print("Saving Grad" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_Grad_rescaled", rescaledGrad)

                        if(args.IGFlag):
                            print("Saving IG" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_IG_rescaled", rescaledIG)



                        if(args.DLFlag):
                            print("Saving DL" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_DL_rescaled", rescaledDL)



                        if(args.GSFlag):
                            print("Saving GS" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_GS_rescaled", rescaledGS)


                        if(args.DLSFlag):
                            print("Saving DLS" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_DLS_rescaled", rescaledDLS)


                        if(args.SGFlag):
                            print("Saving SG" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_SG_rescaled", rescaledSG)

                        if(args.ShapleySamplingFlag):
                            print("Saving ShapleySampling" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_ShapleySampling_rescaled", rescaledShapleySampling)


                        if(args.FeaturePermutationFlag):
                            print("Saving FeaturePermutation" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_FeaturePermutation_rescaled", rescaledFeaturePermutation)


                        if(args.FeatureAblationFlag):
                            print("Saving FeatureAblation" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_FeatureAblation_rescaled", rescaledFeatureAblation)


                        if(args.OcclusionFlag):
                            print("Saving Occlusion" ,modelName+"_"+models[m])
                            np.save(args.Results_dir+modelName+"_"+models[m]+"_Occlusion_rescaled", rescaledOcclusion)

                else:
                    logging.basicConfig(filename=args.log_file,level=logging.DEBUG)

                    logging.debug('{} {} model BestAcc {:.4f}'.format(args.DataName,models[m],Test_Acc))

                    if not os.path.exists(args.ignore_list):
                        with open(args.ignore_list, 'w') as fp: 
                            fp.write(args.DataName+'_'+models[m]+'\n')

                    else:
                        with open(args.ignore_list, "a") as fp:
                            fp.write(args.DataName+'_'+models[m]+'\n')
                    
     

def parse_arguments(argv):


    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default="../Datasets/")
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--logging', type=bool, default=True)


    parser.add_argument('--Graph_dir', type=str, default='../Graphs/')
    parser.add_argument('--Results_dir', type=str, default='../Results/Saliency_Values/')
    parser.add_argument('--log_file', type=str, default='bad_model_acc.log')
    parser.add_argument('--ignore_list', type=str, default='ignore_list.txt')




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



    parser.add_argument('--GradFlag', type=bool, default=False)
    parser.add_argument('--IGFlag', type=bool, default=False)
    parser.add_argument('--DLFlag', type=bool, default=False)
    parser.add_argument('--GSFlag', type=bool, default=False)
    parser.add_argument('--DLSFlag', type=bool, default=False)
    parser.add_argument('--SGFlag', type=bool, default=False)
    parser.add_argument('--ShapleySamplingFlag', type=bool, default=False)
    parser.add_argument('--FeaturePermutationFlag', type=bool, default=False)
    parser.add_argument('--FeatureAblationFlag', type=bool, default=True)
    parser.add_argument('--OcclusionFlag', type=bool, default=False)






    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


