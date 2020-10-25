import sys
import argparse
import Helper
import numpy as np
import time

import random


import os






DatasetsTypes= ["Moving_SmallMiddle","Middle","SmallMiddle", "Moving_Middle",  "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]
DataGenerationTypes=[None,"Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ]

models=["LSTM" ,"LSTMWithInputCellAttention","TCN","Transformer"]

percentages=[ i for i in range(10,91,10)]

def main(args):
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

			if(DataGenerationTypes[y]==None):
				args.DataName=DatasetsTypes[x]+"_Box"
			else:
				args.DataName=DatasetsTypes[x]+"_"+DataGenerationTypes[y]

			modelName="Simulated"
			modelName+=args.DataName

			for m in range(len(models)):
				if(args.DataName+"_"+models[m] in ignore_list):
					print("ignoring",args.DataName+"_"+models[m]  )
					continue
				else:

				

					start=time.time()
					for saliency in Saliency_Methods:
						if(saliency!="Random"):
							saliency_= np.load(args.Saliency_dir+modelName+"_"+models[m]+"_"+saliency+"_rescaled.npy")

						else:
							randomSaliencyIndex= random.randint(0,len(Saliency_Methods)-2)
							saliency_= np.load(args.Saliency_dir+modelName+"_"+models[m]+"_"+Saliency_Methods[randomSaliencyIndex]+"_rescaled.npy")
							np.random.shuffle(np.transpose(saliency_))
							np.save(args.Saliency_dir+modelName+"_"+models[m]+"_"+saliency+"_rescaled",saliency_)


						saliency_=saliency_.reshape(saliency_.shape[0],-1)
						indexGrid=np.zeros((saliency_.shape[0],saliency_.shape[1],len(percentages)),dtype='object')
						indexGrid[:,:,:]=np.nan
						for i in range(saliency_.shape[0]):
							indexes = Helper.getIndexOfAllhighestSalientValues(saliency_[i,:],percentages)
							for l in range(len(indexes)):
								indexGrid[i,:len(indexes[l]),l]=indexes[l]
						for p,percentage in enumerate(percentages):
							np.save(args.Mask_dir+modelName+"_"+models[m]+"_"+saliency+"_"+str(percentage)+"_percentSal_rescaled",indexGrid[:,:,p])

					end=time.time()
					print(modelName+"_"+models[m],"time",end-start)
def parse_arguments(argv):

	parser = argparse.ArgumentParser()
	
	parser.add_argument('--NumTimeSteps',type=int,default=50)
	parser.add_argument('--NumFeatures',type=int,default=50)
	parser.add_argument('--Saliency_dir', type=str, default='../Results/Saliency_Values/')

	parser.add_argument('--Mask_dir', type=str, default='../Results/Saliency_Masks/')
	parser.add_argument('--ignore_list', type=str, default='ignore_list.txt')


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