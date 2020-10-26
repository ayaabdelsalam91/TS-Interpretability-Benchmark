import torch
import sys
import argparse
from createDatasets import createDatasets
from train_models import main as train_models
from interpret import main as interpret
from createMasks import main as createMasks 
from getMaskedAccuracy import main as getMaskedAccuracy
from getPrecisionRecall import main as getPrecisionRecall
from getAccuracyMetrics import main as getAccuracyMetrics
from getFeatureTimePrecisionRecall import main as getFeatureTimePrecisionRecall
import timesynth as ts
import numpy as np





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



DatasetsTypes= ["Middle", "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]

ImpTimeSteps=[30,14,30,15,6,6, 40,40,20,20]
ImpFeatures=[30,14,30,15,40,40,6,6,20,20]

StartImpTimeSteps=[10,18,10,18,22,22,5,5,None,None ]
StartImpFeatures=[10,18,10,18,5,5,22,22,None,None ]

Loc1=[None,None,None,None,None,None,None,None,1,1]
Loc2=[None,None,None,None,None,None,None,None,29,29]


FreezeType = [None,None,None,None,None,None,None,None,"Feature","Time"]
isMoving=[False,False,True,True,False,True,False,True,None,None]
isPositional=[False,False,False,False,False,False,False,False,True,True]

DataGenerationTypes=[None ,"Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ]

models=["Transformer" ,"LSTMWithInputCellAttention","TCN","LSTM"]


def main(args):
	# Creating Datasets
	createDatasets(args,DatasetsTypes,ImpTimeSteps,ImpFeatures,StartImpTimeSteps,StartImpFeatures,Loc1,Loc2,FreezeType,isMoving,isPositional,DataGenerationTypes)

	#Train Models
	train_models(args,DatasetsTypes,DataGenerationTypes,models,device)


	#Decreasing batch size for captum 
	args.batch_size=10

	#Get Saliency maps
	interpret(args,DatasetsTypes,DataGenerationTypes,models,device)

	#create Masks
	createMasks(args,DatasetsTypes,DataGenerationTypes,models)

	#Get Masked Accuracy
	getMaskedAccuracy(args,DatasetsTypes,DataGenerationTypes,models,device)


	#Get precsion and recall
	getPrecisionRecall(args,DatasetsTypes,DataGenerationTypes,models)


	#Get AUC, AUR, AUP and AUPR
	getAccuracyMetrics(args,DatasetsTypes,DataGenerationTypes,models)

	#For Feature  and time level precsion and recall
	getFeatureTimePrecisionRecall(args,DatasetsTypes,DataGenerationTypes,models)
	args.Feature_PrecisionRecall=True
	args.Time_PrecisionRecall=True
	getAccuracyMetrics(args,DatasetsTypes,DataGenerationTypes,models)






def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--DataName', type=str, default="MiddleBox")
	parser.add_argument('--DataGenerationProcess', type=str, default=None)

	parser.add_argument('--plot', type=bool, default=True)
	parser.add_argument('--save', type=bool, default=True)

	parser.add_argument('--isMoving', type=bool, default=False)
	parser.add_argument('--isPositional', type=bool, default=False)


	parser.add_argument('--Sampler', type=str, default="irregular")
	parser.add_argument('--hasNoise', type=bool, default=True)
	parser.add_argument('--Kernal', type=str, default="Matern")

	parser.add_argument('--Frequency',type=float,default=2.0)
	parser.add_argument('--ar_param',type=float,default=0.9)
	parser.add_argument('--Order',type=int,default=10)


	parser.add_argument('--NumTrainingSamples',type=int,default=1000)
	parser.add_argument('--NumTestingSamples',type=int,default=100)
	parser.add_argument('--ImpTimeSteps',type=int,default=40)
	parser.add_argument('--ImpFeatures',type=int,default=40)
	parser.add_argument('--StartImpTimeSteps',type=int,default=30)
	parser.add_argument('--StartImpFeatures',type=int,default=30)
	parser.add_argument('--FreezeType', type=str, default="Feature")
	parser.add_argument('--Loc1',type=int,default=0)
	parser.add_argument('--Loc2',type=int,default=100)

	parser.add_argument('--Graph_dir', type=str, default='../Graphs/')
	parser.add_argument('--datasets_graphs_dir', type=str, default='../Graphs/Datasets/')
	parser.add_argument('--Saliency_Maps_graphs_dir', type=str, default='../Graphs/Saliency_Maps/')
	
	parser.add_argument('--data_dir', type=str, default="../Datasets/")
	parser.add_argument('--Saliency_dir', type=str, default='../Results/Saliency_Values/')
	parser.add_argument('--Mask_dir', type=str, default='../Results/Saliency_Masks/')
	parser.add_argument('--Masked_Acc_dir', type=str, default= "../Results/Masked_Accuracy/")
	parser.add_argument('--Precision_Recall_dir', type=str, default='../Results/Precision_Recall/')
	parser.add_argument('--Acc_Metrics_dir', type=str, default='../Results/Accuracy_Metrics/')

	parser.add_argument('--num_classes', type=int, default=2)
	parser.add_argument('--NumTimeSteps',type=int,default=50)
	parser.add_argument('--NumFeatures',type=int,default=50)
	parser.add_argument('--d_a', type=int, default=50)
	parser.add_argument('--attention_hops', type=int, default=10)
	parser.add_argument('--n_layers', type=int, default=6)
	parser.add_argument('--heads', type=int, default=5)
	parser.add_argument('--kernel_size', type=int, default=4)
	parser.add_argument('--levels', type=int,default=3)
	parser.add_argument('--hidden_size', type=int,default=5)
	parser.add_argument('--batch_size', type=int,default=50)
	parser.add_argument('--num_epochs', type=int,default=500)
	parser.add_argument('--learning_rate', type=float,default=0.001)
	parser.add_argument('--rnndropout', type=float,default=0.1)


	parser.add_argument('--logging', type=bool, default=True)
	parser.add_argument('--log_file', type=str, default='bad_model_acc.log')
	parser.add_argument('--ignore_list', type=str, default='ignore_list.txt')


	parser.add_argument('--Feature_PrecisionRecall', type=bool, default=False)
	parser.add_argument('--Time_PrecisionRecall', type=bool, default=False)


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
