import sys
import argparse
import Helper
import numpy as np
import time
from numpy import trapz
import pandas as pd
import os










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


    precsionTableCol=['AUP','models','Saliency_Methods','Datasets']
    recallTableCol=['AUR','models','Saliency_Methods','Datasets']
    AUPRTableCol=['AUPR','models','Saliency_Methods','Datasets']
    AUCTableCol=['AUC','models','Saliency_Methods','Datasets']
    SummaryTableCol=['Datasets','Saliency_Methods','AUP','AUR','AUPR','AUC']







    for m,model in enumerate(models):
        print("Metrics for", model)
        precsionTable=[]
        recallTable=[]
        AUPRTable=[]

        if(args.Time_PrecisionRecall):
            time_precsionTable=[]
            time_recallTable=[]
            time_AUPRTable=[]

        if(args.Feature_PrecisionRecall):
            feature_precsionTable=[]
            feature_recallTable=[]
            feature_AUPRTable=[]


        AUCTable=[]
        SummaryTable=[]



        for x in range(len(DatasetsTypes)):
            for y in range(len(DataGenerationTypes)):
                args.DataGenerationProcess=DataGenerationTypes[y]
                if(DataGenerationTypes[y]==None):
                    args.DataName=DatasetsTypes[x]+"_Box"
                else:
                    args.DataName=DatasetsTypes[x]+"_"+DataGenerationTypes[y]



                if(args.DataName+"_"+models[m] in ignore_list):
                    print("ignoring",args.DataName+"_"+models[m]  )
                    continue
                
                else:

                    precision_File=args.Precision_Recall_dir+"Precision_"+ args.DataName+"_"+models[m]+"_rescaled.csv"
                    recall_File=args.Precision_Recall_dir+"/Recall_"+ args.DataName+"_"+models[m]+"_rescaled.csv"
                    precision = Helper.load_CSV(precision_File)[:,1:]
                    recall = Helper.load_CSV(recall_File)[:,1:]

                    Accuracy_File=args.Masked_Acc_dir + args.DataName+"_"+models[m]+"_0_10_20_30_40_50_60_70_80_90_100_percentSal_rescaled.csv"
                    accuracy=Helper.load_CSV(Accuracy_File)[:,1:]


                    if(args.Feature_PrecisionRecall):
                        precision_File_Feature=args.Precision_Recall_dir+"Precision_"+ args.DataName+"_"+models[m]+"_Feature_rescaled.csv"
                        recall_File_Feature=args.Precision_Recall_dir+"/Recall_"+ args.DataName+"_"+models[m]+"_Feature_rescaled.csv"
                        feature_precision = Helper.load_CSV(precision_File_Feature)[:,1:]
                        feature_recall = Helper.load_CSV(recall_File_Feature)[:,1:]

                    if(args.Time_PrecisionRecall):

                        precision_File_Time=args.Precision_Recall_dir+"Precision_"+ args.DataName+"_"+models[m]+"_Time_rescaled.csv"
                        recall_File_Time=args.Precision_Recall_dir+"/Recall_"+ args.DataName+"_"+models[m]+"_Time_rescaled.csv"
                        time_precision = Helper.load_CSV(precision_File_Time)[:,1:]
                        time_recall = Helper.load_CSV(recall_File_Time)[:,1:]
               
                    for i in range(len(Saliency_Methods)):
                        precision_row=[]
                        recall_row=[]

                        if(args.Time_PrecisionRecall):
                            time_precision_row=[]
                            time_recall_row=[]

                        if(args.Feature_PrecisionRecall):
                            feature_precision_row=[]
                            feature_recall_row=[]
                        

                        a=[]
                        b=0.1
                        for j in range(precision.shape[1]):
                            if(not pd.isna(precision[i,j])):
                                precision_row.append(precision[i,j])
                                recall_row.append(recall[i,j])


                                if(args.Time_PrecisionRecall):
                                    time_precision_row.append(time_precision[i,j])
                                    time_recall_row.append(time_recall[i,j])

                                if(args.Feature_PrecisionRecall):
                                    feature_precision_row.append(feature_precision[i,j])
                                    feature_recall_row.append(feature_recall[i,j])


                               
                                a.append(b)
                                b+=0.1

                        
                        AUP= np.trapz(precision_row,x=a)
                        AUR= np.trapz(recall_row,x=a)
                        index_ = np.argsort(recall_row)
                        precision_row = np.array(precision_row)
                        recall_row = np.array(recall_row)
                        AUPR=np.trapz(precision_row[index_],x=recall_row[index_])
                        AUPRTable.append([AUPR,models[m],Saliency_Methods[i],args.DataName])
                        precsionTable.append([AUP,models[m],Saliency_Methods[i],args.DataName])
                        recallTable.append([AUR,models[m],Saliency_Methods[i],args.DataName])

                        if(args.Time_PrecisionRecall):

                            time_AUP= np.trapz(time_precision_row,x=a)
                            time_AUR= np.trapz(time_recall_row,x=a)
                            index_ = np.argsort(time_recall_row)
                            time_precision_row = np.array(time_precision_row)
                            time_recall_row = np.array(time_recall_row)
                            time_AUPR=np.trapz(time_precision_row[index_],x=time_recall_row[index_])
                            time_AUPRTable.append([time_AUPR,models[m],Saliency_Methods[i],args.DataName])
                            time_precsionTable.append([time_AUP,models[m],Saliency_Methods[i],args.DataName])
                            time_recallTable.append([time_AUR,models[m],Saliency_Methods[i],args.DataName])

                        if(args.Feature_PrecisionRecall):
                            
                            feature_AUP= np.trapz(feature_precision_row,x=a)
                            feature_AUR= np.trapz(feature_recall_row,x=a)
                            index_ = np.argsort(feature_recall_row)
                            feature_precision_row = np.array(feature_precision_row)
                            feature_recall_row = np.array(feature_recall_row)
                            feature_AUPR=np.trapz(feature_precision_row[index_],x=feature_recall_row[index_])
                            feature_AUPRTable.append([feature_AUPR,models[m],Saliency_Methods[i],args.DataName])
                            feature_precsionTable.append([feature_AUP,models[m],Saliency_Methods[i],args.DataName])
                            feature_recallTable.append([feature_AUR,models[m],Saliency_Methods[i],args.DataName])
                   





                        accuracy_row=[]
                        a=[]
                        b=0.1
                        for j in range(accuracy.shape[1]):
                            if(not pd.isna(accuracy[i,j])):
                                accuracy_row.append(accuracy[i,j])
                                a.append(b)
                                b+=0.1
                        AUC= np.trapz(accuracy_row,x=a)
                        AUCTable.append([AUC,models[m],Saliency_Methods[i],args.DataName])
                        SummaryTable.append([args.DataName,Saliency_Methods[i],AUP,AUR,AUPR,AUC])







        precsionTable=np.array(precsionTable)
        recallTable=np.array(recallTable)
        AUPRTable=np.array(AUPRTable)
        AUCTable=np.array(AUCTable)
        SummaryTable=np.array(SummaryTable)


        Helper.save_intoCSV(precsionTable,args.Acc_Metrics_dir+model+"_AUP_rescaled.csv",col=precsionTableCol)
        Helper.save_intoCSV(recallTable,args.Acc_Metrics_dir+model+"_AUR_rescaled.csv",col=recallTableCol)
        Helper.save_intoCSV(AUPRTable,args.Acc_Metrics_dir+model+"_AUPR.csv",col=AUPRTableCol)
        Helper.save_intoCSV(AUCTable,args.Acc_Metrics_dir+model+"_AUC.csv",col=AUCTableCol)
        Helper.save_intoCSV(SummaryTable,args.Acc_Metrics_dir+model+"_preformance_summary"+".csv",col=SummaryTableCol)



        if(args.Time_PrecisionRecall):

            time_precsionTable=np.array(time_precsionTable)
            time_recallTable=np.array(time_recallTable)
            time_AUPRTable=np.array(time_AUPRTable)


            Helper.save_intoCSV(time_precsionTable,args.Acc_Metrics_dir+model+"_AUP_Time_rescaled.csv",col=precsionTableCol)
            Helper.save_intoCSV(time_recallTable,args.Acc_Metrics_dir+model+"_AUR_Time_rescaled.csv",col=recallTableCol)
            Helper.save_intoCSV(time_AUPRTable,args.Acc_Metrics_dir+model+"_AUPR_Time_rescaled.csv",col=AUPRTableCol)


        if(args.Feature_PrecisionRecall):

            feature_precsionTable=np.array(feature_precsionTable)
            feature_recallTable=np.array(feature_recallTable)
            feature_AUPRTable=np.array(feature_AUPRTable)



            Helper.save_intoCSV(feature_precsionTable,args.Acc_Metrics_dir+model+"_AUP_Feature_rescaled.csv",col=precsionTableCol)
            Helper.save_intoCSV(feature_recallTable,args.Acc_Metrics_dir+model+"_AUR_Feature_rescaled.csv",col=recallTableCol)
            Helper.save_intoCSV(feature_AUPRTable,args.Acc_Metrics_dir+model+"_AUPR_Feature_rescaled.csv",col=AUPRTableCol)


