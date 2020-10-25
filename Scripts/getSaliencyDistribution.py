import sys
import argparse
import Helper
import numpy as np
import time




ignore_list = ["Moving_SmallMiddle_AutoRegressive_LSTM",\
"RareTime_CAR_LSTM",\
"Moving_RareFeature_Box_TCN",\
"Moving_RareFeature_GaussianProcess_TCN",\
"Moving_SmallMiddle_GaussianProcess_Transformer",\
"Moving_SmallMiddle_AutoRegressive_Transformer",\
"Moving_RareTime_GaussianProcess_Transformer",\
"Moving_RareTime_AutoRegressive_Transformer",\
"Moving_RareFeature_GaussianProcess_Transformer",\
"Moving_RareFeature_AutoRegressive_Transformer"]




DatasetsTypes= ["Moving_SmallMiddle","Middle","SmallMiddle", "Moving_Middle",  "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]
DataGenerationTypes=[None,"Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ]

models=["LSTMWithInputCellAttention" ,"TCN","Transformer","LSTM"]



maskedPercentages=[ i for i in range(0,101,10)]



def main(args):

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

    # Saliency_Methods.append("Random")

    for x in range(len(DatasetsTypes)):
        for y in range(len(DataGenerationTypes)):
            args.DataGenerationProcess=DataGenerationTypes[y]
            if(DataGenerationTypes[y]==None):
                args.DataName=DatasetsTypes[x]+"_Box"
            else:
                args.DataName=DatasetsTypes[x]+"_"+DataGenerationTypes[y]

            modelName="Simulated"
            modelName+=args.DataName

            for m in range(len(models)):
                start = time.time()
                resultFileName=args.Saliency_Distribution_dir + args.DataName+"_"+models[m]



                Y_DimOfGrid=args.NumFeatures*args.NumTimeSteps+1
                X_DimOfGrid=len(Saliency_Methods)
                Grid = np.zeros((X_DimOfGrid,Y_DimOfGrid),dtype='object')
                Grid[:,0]=Saliency_Methods

                start = time.time()





                if(args.DataName+"_"+models[m] in ignore_list):
                    print("ignoring",args.DataName+"_"+models[m]  )
                    continue

                else:


                    for _, saliency in enumerate(Saliency_Methods):

                        saliencyValues= np.load(args.Saliency_dir+modelName+"_"+models[m]+"_"+saliency+"_rescaled.npy")
                        saliencyValues=saliencyValues.reshape(-1,args.NumFeatures*args.NumTimeSteps)
                    

                        OnOffGrid=np.zeros((saliencyValues.shape[1]))
                        count=0
                        for j in range(saliencyValues.shape[0]):
                        	if(np.max(saliencyValues[j])>0):
                           	
	                            sorted=saliencyValues[j,:].argsort()[::-1]
	                            count+=1
	                            for i in range(sorted.shape[0]):
	                                OnOffGrid[i]+=saliencyValues[j,sorted[i]]

                        OnOffGrid=OnOffGrid/count
                        Grid[_][1:]=OnOffGrid
                        
                    end = time.time()
                    print('{} {} time: {}'.format(args.DataName,models[m],end-start))


                    resultFileName=resultFileName+"_plotAll_rescaled.csv"
                    Helper.save_intoCSV(Grid,resultFileName)


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--NumTimeSteps',type=int,default=50)
    parser.add_argument('--NumFeatures',type=int,default=50)
    parser.add_argument('--DataGenerationProcess', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default="../Datasets/")
    parser.add_argument('--Mask_dir', type=str, default='../Results/Saliency_Masks/')
    parser.add_argument('--Masked_Acc_dir', type=str, default= "../Results/Masked_Accuracy/")
    parser.add_argument('--Saliency_Distribution_dir', type=str, default= "../Results/Saliency_Distribution/")
    parser.add_argument('--Saliency_dir', type=str, default='../Results/Saliency_Values/')

    parser.add_argument('--Sampler', type=str, default="irregular")
    parser.add_argument('--hasNoise', type=bool, default=True)
    parser.add_argument('--Kernal', type=str, default="Matern")
    parser.add_argument('--Frequency',type=float,default=2.0)
    parser.add_argument('--ar_param',type=float,default=0.9)
    parser.add_argument('--Order',type=int,default=10)


    parser.add_argument('--batch_size', type=int,default=10)

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

    parser.add_argument('--plot', type=bool, default=True)

    parser.add_argument('--Graph_dir', type=str, default='../Graphs/')

    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))