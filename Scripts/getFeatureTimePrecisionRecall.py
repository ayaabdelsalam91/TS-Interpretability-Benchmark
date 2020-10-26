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
        
            referenceIndxAll_Time,referenceIndxAll_Features=Helper.getRowColMaskIndex(referenceIndxAll,args.NumTimeSteps,args.NumFeatures)

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


                    precision_time=np.zeros((len(Saliency_Methods),len(maskedPercentages)),dtype=object)
                    precision_time[:,0]=Saliency_Methods
                    recall_time=np.copy(precision_time)
                    precision_features=np.copy(precision_time)
                    recall_features=np.copy(precision_time)


                    start=time.time()


                    for s,saliency in enumerate(Saliency_Methods):

                        start=time.time()


                        timePrecision=[]
                        timeRecall=[]

                        featurePrecision=[]
                        featureRecall=[]
                        saliencyValues= np.load(args.Saliency_dir+modelName+"_"+models[m]+"_"+saliency+"_rescaled.npy")


                        saliencyValues_time_AverageOfMaxX=np.zeros((saliencyValues.shape[0],args.NumTimeSteps))
                        saliencyValues_feature_AverageOfMaxX=np.zeros((saliencyValues.shape[0],args.NumFeatures))

                        for d in range(saliencyValues.shape[0]):
                            for r in range (args.NumTimeSteps):
                                saliencyValues_time_AverageOfMaxX[d,r]=Helper.getAverageOfMaxX(saliencyValues[d,r,:],int(TargetTS_Ends[d]-TargetTS_Starts[d]))

                                
                            for c in range(args.NumFeatures):
                                saliencyValues_feature_AverageOfMaxX[d,c]=Helper.getAverageOfMaxX(saliencyValues[d,:,c],int(TargetFeat_Ends[d]-TargetFeat_Starts[d]))

                        for maskNumber in range(0,100,10):
                            timeOverallRecall=0
                            timeOverallPrecision=0

                            featureOverallRecall=0
                            featureOverallPrecision=0

                            if(maskNumber !=100 and maskNumber !=0 ):
                                mask = np.load(args.Mask_dir+modelName+"_"+models[m]+"_"+saliency+"_"+str(maskNumber)+"_percentSal_rescaled.npy")

                                maskTime,maskFeatures=Helper.getRowColMaskIndex(mask,args.NumTimeSteps,args.NumFeatures)

                                TimeRcout=0
                                TimePcount=0
                                FeatureRcout=0
                                FeaturePcount=0
                                for i in range(mask.shape[0]):
                                    TP=0
                                    FP=0
                                    FN=0
                                    #For time
                                    for t in range(args.NumTimeSteps):
                                        if(referenceIndxAll_Time[i,t] and maskTime[i,t]):
                                            TP+=saliencyValues_time_AverageOfMaxX[i,t]
                                        elif((not referenceIndxAll_Time[i,t]) and maskTime[i,t]):
                                            FP+=saliencyValues_time_AverageOfMaxX[i,t]
                                        elif(referenceIndxAll_Time[i,t] and  (not maskTime[i,t])):
                                            FN+=saliencyValues_time_AverageOfMaxX[i,t]



                                    if((TP+FP)>0):
                                        timeExamplePrecision=TP/(TP+FP)
                                        TimePcount+=1
                                    else:
                                        timeExamplePrecision=0
                                    if((TP+FN)>0):
                                        timeExampleRecall=TP/(TP+FN)
                                        TimeRcout+=1
                                    else:
                                        timeExampleRecall=0

                                    timeOverallPrecision+=timeExamplePrecision
                                    timeOverallRecall+=timeExampleRecall

                                    TP=0
                                    FP=0
                                    FN=0
                                    #For Feature
                                    for f in range(args.NumFeatures):
                                        if(referenceIndxAll_Features[i,f] and maskFeatures[i,f]):
                                            TP+=saliencyValues_feature_AverageOfMaxX[i,f]
                                        elif((not referenceIndxAll_Features[i,f]) and maskFeatures[i,f]):
                                            FP+=saliencyValues_feature_AverageOfMaxX[i,f]
                                        elif(referenceIndxAll_Features[i,f] and  (not maskFeatures[i,f])):
                                            FN+=saliencyValues_feature_AverageOfMaxX[i,f]

                                    if((TP+FP)>0):
                                        featureExamplePrecision=TP/(TP+FP)
                                        FeaturePcount+=1
                                    else:
                                        featureExamplePrecision=0
                                    if((TP+FN)>0):
                                        featureExampleRecall=TP/(TP+FN)
                                        FeatureRcout+=1
                                    else:
                                        featureExampleRecall=0

                                    featureOverallPrecision+=featureExamplePrecision
                                    featureOverallRecall+=featureExampleRecall




                                timeOverallPrecision=timeOverallPrecision/TimePcount
                                timeOverallRecall=timeOverallRecall/TimeRcout
                                timePrecision.append(timeOverallPrecision)
                                timeRecall.append(timeOverallRecall)

                                featureOverallPrecision=featureOverallPrecision/FeaturePcount
                                featureOverallRecall=featureOverallRecall/FeatureRcout
                                featurePrecision.append(featureOverallPrecision)
                                featureRecall.append(featureOverallRecall)
                            else:
                                featurePrecision.append(np.nan)
                                featureRecall.append(np.nan)
                                timePrecision.append(np.nan)
                                timeRecall.append(np.nan)


                            print('{} {} {} masked percentages {} Feature Precision {:.4f} Feature Recall {:.4f} Time Precision {:.4f} Time Recall {:.4f}'.format(args.DataName,models[m],saliency,maskNumber,featureOverallPrecision,featureOverallRecall,timeOverallPrecision,timeOverallRecall))

                        precision_time[s,1:]=timePrecision
                        recall_time[s,1:]=timeRecall

                        precision_features[s,1:]=featurePrecision
                        recall_features[s,1:]=featureRecall
                    end=time.time()
                    print(args.DataName+"_"+models[m],end-start)


                    
                    precision_File=args.Precision_Recall_dir+"Precision_"+ args.DataName+"_"+models[m]+"_Feature_rescaled.csv"
                    recall_File=args.Precision_Recall_dir+"/Recall_"+ args.DataName+"_"+models[m]+"_Feature_rescaled.csv"

                    Helper.save_intoCSV(precision_features,precision_File,col=cols)
                    Helper.save_intoCSV(recall_features,recall_File,col=cols)


                    precision_File=args.Precision_Recall_dir+"Precision_"+ args.DataName+"_"+models[m]+"_Time_rescaled.csv"
                    recall_File=args.Precision_Recall_dir+"/Recall_"+ args.DataName+"_"+models[m]+"_Time_rescaled.csv"
                    Helper.save_intoCSV(precision_time,precision_File,col=cols)
                    Helper.save_intoCSV(recall_time,recall_File,col=cols)


            boxEndTime=time.time()
            print(args.DataName,boxEndTime-boxStartTime)
            print()

            np.load = np_load_old


