import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import Helper
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from Helper import checkAccuracy 
sys.path.append('./Models/')
from LSTM import LSTM
from Transformer import Transformer
from LSTMWithInputCellAttention import LSTMWithInputCellAttention
from TCN import TCN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first =True

DatasetsTypes= ["Middle", "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]
DataGenerationTypes=[None,"Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ]

DatasetsTypes= ["Moving_SmallMiddle","Moving_RareTime","Moving_RareFeature"]
DataGenerationTypes=["AutoRegressive", "GaussianProcess"]
models=["LSTM","LSTMWithInputCellAttention","TCN","Transformer"]
# SmallMiddle_Harmonic
# DatasetsTypes= ["Middle"]#, "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"]
# DataGenerationTypes=[None]#,"Harmonic", "GaussianProcess", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ]
# Moving_SmallMiddle_AutoRegressive
models=["Transformer"]#,"LSTMWithInputCellAttention","TCN","Transformer"]


def main(args):
	criterion = nn.CrossEntropyLoss()
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
				print(Training.shape)

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


				if(models[m]=="LSTM"):
					net=LSTM(args.NumFeatures, args.hidden_size,args.num_classes,args.rnndropout).to(device)
				elif(models[m]=="LSTMWithInputCellAttention"):
					net=LSTMWithInputCellAttention(args.NumFeatures, args.hidden_size,args.num_classes,args.rnndropout,args.attention_hops,args.d_a).to(device)
				elif(models[m]=="Transformer"):
					net=Transformer(args.NumFeatures, args.NumTimeSteps, args.n_layers, args.heads, args.rnndropout,args.num_classes,time=args.NumTimeSteps).to(device)
				elif(models[m]=="TCN"):
					num_chans = [args.hidden_size] * (args.levels - 1) + [args.NumTimeSteps]
					net=TCN(args.NumFeatures,args.num_classes,num_chans,args.kernel_size,args.rnndropout,time=args.NumTimeSteps).to(device)

				net.double()
				optimizerTimeAtten = torch.optim.Adam(net.parameters(), lr=args.learning_rate)


				saveModelName="../Models/"+models[m]+"/"+modelName
				saveModelBestName =saveModelName +"_BEST.pkl"
				saveModelLastName=saveModelName+"_LAST.pkl"






				total_step = len(train_loaderRNN)
				Train_acc_flag=False
				Train_Acc=0
				Test_Acc=0
				BestAcc=0
				BestEpochs = 0
				patience=200

				for epoch in range(args.num_epochs):
					noImprovementflag=True
					for i, (samples, labels) in enumerate(train_loaderRNN):

						net.train()
						samples = samples.reshape(-1, args.NumTimeSteps, args.NumFeatures).to(device)
						samples = Variable(samples)
						labels = labels.to(device)
						labels = Variable(labels).long()

						outputs = net(samples)
						loss = criterion(outputs, labels)

						optimizerTimeAtten.zero_grad()
						loss.backward()
						optimizerTimeAtten.step()

						if (i+1) % 3 == 0:
							Test_Acc = checkAccuracy(test_loaderRNN, net,args)
							Train_Acc = checkAccuracy(train_loaderRNN, net,args)
							if(Test_Acc>BestAcc):
								BestAcc=Test_Acc
								BestEpochs = epoch+1
								torch.save(net, saveModelBestName)
								noImprovementflag=False

						print ('{} {}-->Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy {:.2f}, Test Accuracy {:.2f},BestEpochs {},BestAcc {:.2f} patience {}' 
						   .format(args.DataName, models[m] ,epoch+1, args.num_epochs, i+1, total_step, loss.item(),Train_Acc, Test_Acc,BestEpochs , BestAcc , patience))

					if(noImprovementflag):
						patience-=1
					else:
						patience=200

					if(epoch+1)%10==0:
						torch.save(net, saveModelLastName)
					if(Train_Acc>=99 or BestAcc>=99 ):
						torch.save(net,saveModelLastName)
						Train_acc_flag=True
						break
					if(Train_acc_flag or patience==0):
						break

					Train_Acc =checkAccuracy(train_loaderRNN , net, args)
					print('{} {} BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(args.DataName, models[m] ,BestEpochs , BestAcc , Train_Acc))



def parse_arguments(argv):

	parser = argparse.ArgumentParser()
	parser.add_argument('--num_classes', type=int, default=2)
	parser.add_argument('--data_dir', type=str, default="../Datasets/")


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
	parser.add_argument('--data-dir', help='Data  directory', action='store', type=str ,default="../Datasets/")
	return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))