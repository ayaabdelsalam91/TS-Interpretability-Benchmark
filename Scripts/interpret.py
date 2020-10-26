import torch
import argparse
import sys
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
import Helper
from Helper import checkAccuracy  
import random
from sklearn.preprocessing import StandardScaler,MinMaxScaler
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





def main(args,DatasetsTypes,DataGenerationTypes,models,device):
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
                        plotExampleBox(TestingRNN[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_sample',flip=True)

                        print("Plotting sample",index)
                        if(args.GradFlag):
                            plotExampleBox(rescaledGrad[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_Grad',greyScale=True,flip=True)

                        if(args.IGFlag):
                            plotExampleBox(rescaledIG[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_IG',greyScale=True, flip=True)


                        if(args.DLFlag):
                            plotExampleBox(rescaledDL[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_DL',greyScale=True, flip=True)


                        if(args.GSFlag):
                            plotExampleBox(rescaledGS[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_GS',greyScale=True, flip=True)


                        if(args.DLSFlag):
                            plotExampleBox(rescaledDLS[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_DLS',greyScale=True, flip=True)


                        if(args.SGFlag):
                            plotExampleBox(rescaledSG[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_SG',greyScale=True, flip=True)


                        if(args.ShapleySamplingFlag):
                            plotExampleBox(rescaledShapleySampling[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_ShapleySampling',greyScale=True, flip=True)


                        if(args.FeaturePermutationFlag):
                            plotExampleBox(rescaledFeaturePermutation[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_FeaturePermutation',greyScale=True, flip=True)


                        if(args.FeatureAblationFlag):
                            plotExampleBox(rescaledFeatureAblation[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_FeatureAblation',greyScale=True, flip=True)

                        if(args.OcclusionFlag):
                            plotExampleBox(rescaledOcclusion[index,:,:],args.Saliency_Maps_graphs_dir+args.DataName+"_"+models[m]+'_Occlusion',greyScale=True, flip=True)

                    if(args.save):
                        if(args.GradFlag):
                            print("Saving Grad" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_Grad_rescaled", rescaledGrad)

                        if(args.IGFlag):
                            print("Saving IG" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_IG_rescaled", rescaledIG)



                        if(args.DLFlag):
                            print("Saving DL" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_DL_rescaled", rescaledDL)



                        if(args.GSFlag):
                            print("Saving GS" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_GS_rescaled", rescaledGS)


                        if(args.DLSFlag):
                            print("Saving DLS" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_DLS_rescaled", rescaledDLS)


                        if(args.SGFlag):
                            print("Saving SG" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_SG_rescaled", rescaledSG)

                        if(args.ShapleySamplingFlag):
                            print("Saving ShapleySampling" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_ShapleySampling_rescaled", rescaledShapleySampling)


                        if(args.FeaturePermutationFlag):
                            print("Saving FeaturePermutation" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_FeaturePermutation_rescaled", rescaledFeaturePermutation)


                        if(args.FeatureAblationFlag):
                            print("Saving FeatureAblation" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_FeatureAblation_rescaled", rescaledFeatureAblation)


                        if(args.OcclusionFlag):
                            print("Saving Occlusion" ,modelName+"_"+models[m])
                            np.save(args.Saliency_dir+modelName+"_"+models[m]+"_Occlusion_rescaled", rescaledOcclusion)

                else:
                    logging.basicConfig(filename=args.log_file,level=logging.DEBUG)

                    logging.debug('{} {} model BestAcc {:.4f}'.format(args.DataName,models[m],Test_Acc))

                    if not os.path.exists(args.ignore_list):
                        with open(args.ignore_list, 'w') as fp: 
                            fp.write(args.DataName+'_'+models[m]+'\n')

                    else:
                        with open(args.ignore_list, "a") as fp:
                            fp.write(args.DataName+'_'+models[m]+'\n')
                    
     
