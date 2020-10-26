import timesynth as ts
import numpy as np
import sys
import argparse
from Plotting import *

def createSample(args,Target,start_ImpTS,end_ImpTS,start_ImpFeat,end_ImpFeat):

	if(args.DataGenerationProcess==None):
		sample=np.random.normal(0,1,[args.NumTimeSteps,args.NumFeatures])
		Features=np.random.normal(Target,1,[args.ImpTimeSteps,args.ImpFeatures])
		sample[start_ImpTS:end_ImpTS,start_ImpFeat:end_ImpFeat]=Features
		# print(start_ImpFeat,end_ImpFeat)

	else:
		time_sampler = ts.TimeSampler(stop_time=20)
		sample=np.zeros([args.NumTimeSteps,args.NumFeatures])


		if(args.Sampler=="regular"):
			time = time_sampler.sample_regular_time(num_points=args.NumTimeSteps*2, keep_percentage=50)
		else:
			time = time_sampler.sample_irregular_time(num_points=args.NumTimeSteps*2, keep_percentage=50)

		
		for  i in range(args.NumFeatures):
			if(args.DataGenerationProcess== "Harmonic"):
				 signal = ts.signals.Sinusoidal(frequency=args.Frequency)
				
			elif(args.DataGenerationProcess=="GaussianProcess"):
				signal = ts.signals.GaussianProcess(kernel=args.Kernal, nu=3./2)

			elif(args.DataGenerationProcess=="PseudoPeriodic"):
				signal = ts.signals.PseudoPeriodic(frequency=args.Frequency, freqSD=0.01, ampSD=0.5)

			elif(args.DataGenerationProcess=="AutoRegressive"):
				signal = ts.signals.AutoRegressive(ar_param=[args.ar_param])

			elif(args.DataGenerationProcess=="CAR"):
				signal = ts.signals.CAR(ar_param=args.ar_param, sigma=0.01)

			elif(args.DataGenerationProcess=="NARMA"):
				signal = ts.signals.NARMA(order=args.Order)

			if(args.hasNoise):
	 			noise= ts.noise.GaussianNoise(std=0.3)
			 	timeseries = ts.TimeSeries(signal, noise_generator=noise)
			else:
			 	timeseries = ts.TimeSeries(signal)

			feature, signals, errors = timeseries.sample(time)
			sample[:,i]= feature



		sample[start_ImpTS:end_ImpTS,start_ImpFeat:end_ImpFeat]=sample[start_ImpTS:end_ImpTS,start_ImpFeat:end_ImpFeat]+Target
	return sample




def createDataset(args,NumberOFsamples):

	DataSet = np.zeros((NumberOFsamples ,args.NumTimeSteps , args.NumFeatures))
	metaData= np.zeros((NumberOFsamples,5))
	Targets = np.random.randint(-1, 1,NumberOFsamples)

	TargetTS_Ends=np.zeros((NumberOFsamples,))
	TargetFeat_Ends=np.zeros((NumberOFsamples,))

	if(args.isMoving):
		TargetTS_Starts = np.random.randint(args.NumTimeSteps-args.ImpTimeSteps, size=NumberOFsamples)
		TargetFeat_Starts = np.random.randint(args.NumFeatures-args.ImpFeatures, size=NumberOFsamples)


	else:
		TargetTS_Starts=np.zeros((NumberOFsamples,))
		TargetFeat_Starts=np.zeros((NumberOFsamples,))

		TargetTS_Starts[:]= args.StartImpTimeSteps 
		TargetFeat_Starts[:]= args.StartImpFeatures


	for i in range (NumberOFsamples):
		if(Targets[i]==0):
			Targets[i]=1

		TargetTS_Ends[i],TargetFeat_Ends[i] = TargetTS_Starts[i]+args.ImpTimeSteps, TargetFeat_Starts[i]+args.ImpFeatures
		sample = createSample(args,Targets[i],int(TargetTS_Starts[i]),int(TargetTS_Ends[i]),int(TargetFeat_Starts[i]),int(TargetFeat_Ends[i]))

		if(Targets[i]==-1):
			Targets[i]=0

		DataSet[i,:,:,]=sample

	#Label
	metaData[:,0]=Targets
	#Start important time
	metaData[:,1]=TargetTS_Starts
	#End important time
	metaData[:,2]=TargetTS_Ends
	#Start important feature
	metaData[:,3]=TargetFeat_Starts
	#End important feature
	metaData[:,4]=TargetFeat_Ends



	return DataSet , metaData









def createPositionalDataset(args,NumberOFsamples):
	DataSet = np.zeros((NumberOFsamples ,args.NumTimeSteps , args.NumFeatures  ))
	metaData= np.zeros((NumberOFsamples,5))
	Targets = np.random.randint(-1, 1,NumberOFsamples)

	TargetTS_Ends=np.zeros((NumberOFsamples,))
	TargetFeat_Ends=np.zeros((NumberOFsamples,))

	if (args.FreezeType=="Feature"):

		TargetTS_Starts = np.random.randint(args.NumTimeSteps-args.ImpTimeSteps, size=NumberOFsamples)		
		TargetFeat_Starts=np.zeros((NumberOFsamples,))

		for i in range (NumberOFsamples):
			if(Targets[i]==0):
				Targets[i]=1
				TargetYStart,TargetXStart = TargetTS_Starts[i], args.Loc1
			else:
				TargetYStart,TargetXStart = TargetTS_Starts[i], args.Loc2

			# print(TargetXStart)
			TargetFeat_Starts[i]=TargetXStart

			TargetYEnd,TargetXEnd = TargetYStart+args.ImpTimeSteps, TargetXStart+args.ImpFeatures

			sample = createSample(args,1,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd)
			if(Targets[i]==-1):
				Targets[i]=0

			TargetTS_Ends[i] = TargetTS_Starts[i]+args.ImpTimeSteps
			TargetFeat_Ends[i] = TargetFeat_Starts[i]+args.ImpFeatures

			DataSet[i,:,:,]=sample

	else:
		TargetFeat_Starts = np.random.randint( args.NumFeatures -args.ImpFeatures, size=NumberOFsamples)
		TargetTS_Starts=np.zeros((NumberOFsamples,))

		for i in range (NumberOFsamples):
			if(Targets[i]==0):
				Targets[i]=1
				TargetYStart,TargetXStart = args.Loc1, TargetFeat_Starts[i]
			else:
				TargetYStart,TargetXStart = args.Loc2, TargetFeat_Starts[i]

			TargetTS_Starts[i]=TargetYStart

			TargetYEnd,TargetXEnd = TargetYStart+args.ImpTimeSteps, TargetXStart+args.ImpFeatures

			sample = createSample(args,1,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd)
			if(Targets[i]==-1):
				Targets[i]=0

			TargetTS_Ends[i] = TargetTS_Starts[i]+args.ImpTimeSteps
			TargetFeat_Ends[i] = TargetFeat_Starts[i]+args.ImpFeatures



			DataSet[i,:,:,]=sample




	#Label
	metaData[:,0]=Targets
	#Start important time
	metaData[:,1]=TargetTS_Starts
	#End important time
	metaData[:,2]=TargetTS_Ends
	#Start important feature
	metaData[:,3]=TargetFeat_Starts
	#End important feature
	metaData[:,4]=TargetFeat_Ends

	return DataSet, metaData




def createSimulationDataProcesses(args):








	if(args.isPositional):
		print("Creating Positional Training Dataset" , args.DataName)
		TrainingDataset  , TrainingDataset_MetaData= createPositionalDataset(args , args.NumTrainingSamples)
		print("Creating Positional Testing Dataset", args.DataName)
		TestingDataset ,TestingDataset_MetaData= createPositionalDataset(args,args.NumTestingSamples)

	else:

		print("Creating Training Dataset", args.DataName)
		TrainingDataset  , TrainingDataset_MetaData= createDataset(args , args.NumTrainingSamples)
		print("Creating Testing Dataset", args.DataName)
		TestingDataset ,TestingDataset_MetaData= createDataset(args,args.NumTestingSamples)





	if(args.plot==True):
		print("Plotting Samples...")
		if(args.isPositional):
			negIndex=[]
			posIndex=[]

			for i in range(TrainingDataset_MetaData.shape[0]):
				if(TrainingDataset_MetaData[i,0]==1 and  len(posIndex)<2):
					posIndex.append(i)
				elif(TrainingDataset_MetaData[i,0]==0 and  len(negIndex)<2):
					negIndex.append(i)

				if(len(negIndex)==2 and len(posIndex)==2):
					break

			if(args.DataGenerationProcess==None):
				plotExampleBox(TrainingDataset[negIndex[0],:,:],args.datasets_graphs_dir+args.DataName+'_negtive1' ,flip=True)
				plotExampleBox(TrainingDataset[posIndex[0],:,:],args.datasets_graphs_dir+args.DataName+'_postive1' ,flip=True)
				
				plotExampleBox(TrainingDataset[negIndex[1],:,:],args.datasets_graphs_dir+args.DataName+'_negtive2' ,flip=True)
				plotExampleBox(TrainingDataset[posIndex[1],:,:],args.datasets_graphs_dir+args.DataName+'_postive2' ,flip=True)


			else:
				plotExampleProcesses(TrainingDataset[negIndex[0],:,:],args.datasets_graphs_dir+args.DataName+'_negtive1')
				plotExampleProcesses(TrainingDataset[posIndex[0],:,:],args.datasets_graphs_dir+args.DataName+'_postive1',color='b')
				plotExampleProcesses(TrainingDataset[negIndex[1],:,:],args.datasets_graphs_dir+args.DataName+'_negtive2')
				plotExampleProcesses(TrainingDataset[posIndex[1],:,:],args.datasets_graphs_dir+args.DataName+'_postive2',color='b')

		else:
			
			negIndex=-1
			posIndex=-1


			for i in range(TrainingDataset_MetaData.shape[0]):
				if(TrainingDataset_MetaData[i,0]==1):
					posIndex=i
					# print(i , TrainingDataset_MetaData[i,:])
				else:
					negIndex=i
					# print(i , TrainingDataset_MetaData[i,:])

				if(negIndex!=-1 and posIndex!=-1):
					break


			plotExampleBox(TrainingDataset[negIndex,:,:],args.datasets_graphs_dir+args.DataName+'_negtive_heatmap',flip=True)
			plotExampleBox(TrainingDataset[posIndex,:,:],args.datasets_graphs_dir+args.DataName+'_postive_heatmap',flip=True)

			plotExampleProcesses(TrainingDataset[negIndex,:,:],args.datasets_graphs_dir+args.DataName+'_negtive_signal')
			plotExampleProcesses(TrainingDataset[posIndex,:,:],args.datasets_graphs_dir+args.DataName+'_postive_signal',color='b')

	if(args.save==True):
		print("Saving Datasets...")
		np.save(args.data_dir+"SimulatedTrainingData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps),TrainingDataset)
		np.save(args.data_dir+"SimulatedTrainingMetaData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps),TrainingDataset_MetaData)

		np.save(args.data_dir+"SimulatedTestingData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps),TestingDataset)
		np.save(args.data_dir+"SimulatedTestingMetaData_"+args.DataName+"_F_"+str(args.NumFeatures)+"_TS_"+str(args.NumTimeSteps),TestingDataset_MetaData)
