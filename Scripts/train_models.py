import sys
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from Helper import checkAccuracy 
sys.path.append('./Models/')
from LSTM import LSTM
from Transformer import Transformer
from LSTMWithInputCellAttention import LSTMWithInputCellAttention
from TCN import TCN
import torch.nn as nn






def main(args,DatasetsTypes,DataGenerationTypes,models,device):
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
						if(Train_Acc>=99 or BestAcc>=99 ):
							torch.save(net,saveModelLastName)
							Train_acc_flag=True
							break

					if(noImprovementflag):
						patience-=1
					else:
						patience=200

					if(epoch+1)%10==0:
						torch.save(net, saveModelLastName)

					if(Train_acc_flag or patience==0):
						break

					Train_Acc =checkAccuracy(train_loaderRNN , net, args)
					print('{} {} BestEpochs {},BestAcc {:.4f}, LastTrainAcc {:.4f},'.format(args.DataName, models[m] ,BestEpochs , BestAcc , Train_Acc))
