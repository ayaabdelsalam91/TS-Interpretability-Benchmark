import sys
import argparse
from createSimulationDataProcesses import main as createSimulationDataProcesses








DatasetsTypes= ["Middle"]#, "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]

ImpTimeSteps=[30,14,30,15,6,6, 40,40,20,20]
ImpFeatures=[30,14,30,15,40,40,6,6,20,20]

StartImpTimeSteps=[10,18,10,18,22,22,5,5,None,None ]
StartImpFeatures=[10,18,10,18,5,5,22,22,None,None ]

Loc1=[None,None,None,None,None,None,None,None,1,1]
Loc2=[None,None,None,None,None,None,None,None,29,29]




# ImpTimeSteps=[50,30,50,30,10,10, 80,80,40,40]
# ImpFeatures=[50,30,50,30,80,80,10,10,40,40]

# StartImpTimeSteps=[25,35,25,35,45,45,10,10,None,None ]
# StartImpFeatures=[25,35,25,35,10,10,45,45,None,None ]

# Loc1=[None,None,None,None,None,None,None,None,1,1]
# Loc2=[None,None,None,None,None,None,None,None,59,59]

FreezeType = [None,None,None,None,None,None,None,None,"Feature","Time"]
isMoving=[False,False,True,True,False,True,False,True,None,None]
isPositional=[False,False,False,False,False,False,False,False,True,True]

DataGenerationTypes=[None,"Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ]




def main(args):

	for i in range(len(DatasetsTypes)):
		if(i==0):

			args.ImpTimeSteps=ImpTimeSteps[i]
			args.ImpFeatures=ImpFeatures[i]

			args.StartImpTimeSteps=StartImpTimeSteps[i]
			args.StartImpFeatures=StartImpFeatures[i]

			args.Loc1=Loc1[i]
			args.Loc2=Loc2[i]

			args.FreezeType=FreezeType[i]
			args.isMoving=isMoving[i]
			args.isPositional=isPositional[i]


			for j in range(len(DataGenerationTypes)):
				if(DataGenerationTypes[j]==None):
					args.DataName=DatasetsTypes[i]+"_Box"
				else:
					args.DataName=DatasetsTypes[i]+"_"+DataGenerationTypes[j]
				args.DataGenerationProcess=DataGenerationTypes[j]

				createSimulationDataProcesses(args)
				break









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


	parser.add_argument('--NumTimeSteps',type=int,default=50)
	parser.add_argument('--NumFeatures',type=int,default=50)


	parser.add_argument('--ImpTimeSteps',type=int,default=40)
	parser.add_argument('--ImpFeatures',type=int,default=40)

	parser.add_argument('--StartImpTimeSteps',type=int,default=30)
	parser.add_argument('--StartImpFeatures',type=int,default=30)



	parser.add_argument('--FreezeType', type=str, default="Feature")
	parser.add_argument('--Loc1',type=int,default=0)
	parser.add_argument('--Loc2',type=int,default=100)



	parser.add_argument('--Graph_dir', type=str, default='../Graphs/')
	parser.add_argument('--data_dir', type=str, default="../Datasets/")


	return  parser.parse_args()

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))