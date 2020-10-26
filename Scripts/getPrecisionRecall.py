import sys
import argparse
import Helper
import numpy as np
import time
import random
import os
import pandas as pd














maskedPercentages=[ i for i in range(0,101,10)]

def main(args,DatasetsTypes,DataGenerationTypes,models):
    if  os.path.exists(args.ignore_list):
        f = open(args.ignore_list, 'r+')
        ignore_list = [line for line in f.readlines()]
        f.close()
        for i in range(len(ignore_list)):
            if('\n' in ignore_list[i]):
                ignore_list[i]=ignore_list[i][:-1]
    else:
        ignore_list=[]

    cols=["Saliency_Methods"]

    for p in range(0,100,10):
        cols.append(str(p))

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
    


            Testing=np.load(args.data_dir+"SimulatedTestingData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
            TestingDataset_MetaData=np.load(args.data_dir+"SimulatedTestingMetaData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps)+".npy")
            TestingLabel=TestingDataset_MetaData[:,0]
            TargetTS_Starts=TestingDataset_MetaData[:,1]
            TargetTS_Ends=TestingDataset_MetaData[:,2]
            TargetFeat_Starts= TestingDataset_MetaData[:,3]
            TargetFeat_Ends= TestingDataset_MetaData[:,4]

            referencesSamples=np.zeros((Testing.shape))
            referenceIndxAll=np.zeros((Testing.shape[0],args.NumTimeSteps*args.NumFeatures))
            referenceIndxAll[:,:]=np.nan


            for i in range(TestingLabel.shape[0]):

                referencesSamples[i,int(TargetTS_Starts[i]):int(TargetTS_Ends[i]),int(TargetFeat_Starts[i]):int(TargetFeat_Ends[i])]=1
                numberOfImpFeatures=int(np.sum(referencesSamples[i,:,:]))
                ind = Helper.getIndexOfXhighestFeatures(referencesSamples[i,:,:].flatten() , numberOfImpFeatures)
                referenceIndxAll[i,:ind.shape[0]]=ind

            modelName="Simulated"
            modelName+=args.DataName

            # save np.load
            np_load_old = np.load

            # modify the default parameters of np.load
            np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

            boxStartTime=time.time()
            for m in range(len(models)):

                if(args.DataName+"_"+models[m] in ignore_list):
                    print("ignoring",args.DataName+"_"+models[m]  )
                    continue

                else:


                    precision_=np.zeros((len(Saliency_Methods),len(maskedPercentages)),dtype=object)
                    precision_[:,0]=Saliency_Methods


                    recall_=np.copy(precision_)
                    start=time.time()
                    for s,saliency in enumerate(Saliency_Methods):

                        precision=[]
                        recall=[]
                        saliencyValues= np.load(args.Saliency_dir+modelName+"_"+models[m]+"_"+saliency+"_rescaled.npy")
                        saliencyValues=saliencyValues.reshape(-1,args.NumFeatures*args.NumTimeSteps)


                        for maskNumber in range(0,100,10):
                            overallRecall=0
                            overallPrecision=0

                            if(maskNumber !=100 and maskNumber !=0 ):
                                mask = np.load(args.Mask_dir+modelName+"_"+models[m]+"_"+saliency+"_"+str(maskNumber)+"_percentSal_rescaled.npy")
                                
                                Rcout=0
                                Pcount=0

                                for i in range(mask.shape[0]):
                                    postiveIndex = mask[i,:]
                                    postiveIndex=postiveIndex[np.logical_not(pd.isna(postiveIndex))]
                                    postiveIndex = postiveIndex.astype(np.int64)

                                    trueIndex=referenceIndxAll[i,:]
                                    trueIndex=trueIndex[np.logical_not(pd.isna(trueIndex))]
                                    trueIndex = trueIndex.astype(np.int64)

                                    postiveWithTrue = np.isin(postiveIndex, trueIndex)
                                    TrueWithpostive = np.isin(trueIndex, postiveIndex)

                                    countTP=0
                                    countFP=0
                                    countFN=0

                                    TP=0
                                    FP=0
                                    FN=0
            
                                    for j in range(postiveWithTrue.shape[0]):
                                        if(postiveWithTrue[j]):
                                            #In postive and true so true postive
                                            TP+=saliencyValues[i,postiveIndex[j]]
                                            countTP+=1
                                        else:
                                            #In postive but not true so false postive
                                            FP+=saliencyValues[i,postiveIndex[j]]
                                            countFP+=1
                                    for j in range(TrueWithpostive.shape[0]):
                                        if(not TrueWithpostive[j]):
                                            #In true but not in postive False negtive
                                            FN+=saliencyValues[i,trueIndex[j]]
                                            countFN+=1


                                    if((TP+FP)>0):
                                        examplePrecision=TP/(TP+FP)
                                        Pcount+=1
                                    else:
                                        examplePrecision=0
                                    if((TP+FN)>0):
                                        exampleRecall=TP/(TP+FN)
                                        Rcout+=1
                                    else:
                                        exampleRecall=0

                                    overallPrecision+=examplePrecision
                                    overallRecall+=exampleRecall

                                overallPrecision=overallPrecision/Pcount
                                overallRecall=overallRecall/Rcout
                                precision.append(overallPrecision)
                                recall.append(overallRecall)
                            else:
                                precision.append(np.nan)
                                recall.append(np.nan)

                            print('{} {} {} masked percentages {} Precision {:.4f} Recall {:.4f}'.format(args.DataName,models[m],saliency,maskNumber,overallPrecision,overallRecall))

                        precision_[s,1:]=precision
                        recall_[s,1:]=recall
                    end=time.time()
                    print(args.DataName+"_"+models[m],end-start)

                    precision_File=args.Precision_Recall_dir+"Precision_"+ args.DataName+"_"+models[m]+"_rescaled.csv"
                    recall_File=args.Precision_Recall_dir+"/Recall_"+ args.DataName+"_"+models[m]+"_rescaled.csv"
                    Helper.save_intoCSV(precision_,precision_File,col=cols)
                    Helper.save_intoCSV(recall_,recall_File,col=cols)
            boxEndTime=time.time()
            print(args.DataName,boxEndTime-boxStartTime)
            print()

            np.load = np_load_old




      
