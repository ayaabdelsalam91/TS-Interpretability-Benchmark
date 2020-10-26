import numpy as np
import pandas as pd
from torch.autograd import Variable
import itertools
import torch
import torch.nn as nn
from  sklearn.preprocessing import minmax_scale
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import torch.nn.functional as F
import sys
sys.path.append('../')
import torch.utils.data as data_utils
import timesynth as ts




################################## General Helper Function ##############################



def load_CSV(file,returnDF=False,Flip=False):
	df = pd.read_csv(file)
	data=df.values
	if(Flip):
		print("Will Un-Flip before Loading")
		data=data.reshape((data.shape[1],data.shape[0]))
	if(returnDF):
		return df
	return data


def givenAttGetRescaledSaliency(args,attributions,isTensor=True):
    if(isTensor):
        saliency = np.absolute(attributions.data.cpu().numpy())
    else:
        saliency = np.absolute(attributions)
    saliency=saliency.reshape(-1,args.NumTimeSteps*args.NumFeatures)
    rescaledSaliency=minmax_scale(saliency,axis=1)
    rescaledSaliency=rescaledSaliency.reshape(attributions.shape)
    return rescaledSaliency



def save_intoCSV(data,file,Flip=False,col=None,index=False):
	if(Flip):
		print("Will Flip before Saving")
		data=data.reshape((data.shape[1],data.shape[0]))


	df = pd.DataFrame(data)
	if(col!=None):
		df.columns = col
	df.to_csv(file,index=index)



def getIndexOfXhighestFeatures(array,X):
    return np.argpartition(array, int(-1*X))[int(-1*X):]


def getAverageOfMaxX(array,X):
    index = getIndexOfXhighestFeatures(array,X)
    avg=np.mean(array[index])
    return avg


def getIndexOfAllhighestSalientValues(array,percentageArray):
    X=array.shape[0]
    # index=np.argpartition(array, int(-1*X))
    index=np.argsort(array)
    totalSaliency=np.sum(array)
    indexes=[]
    X=1
    for percentage in percentageArray:
        actualPercentage=percentage/100
        
        index_X=index[int(-1*X):]

        percentageDroped=np.sum(array[index_X])/totalSaliency
        if(percentageDroped<actualPercentage):
            X_=X+1
            index_X_=index[int(-1*X_):]
            percentageDroped_=np.sum(array[index_X])/totalSaliency
            if(not (percentageDroped_>actualPercentage)):
                while(percentageDroped<actualPercentage and X<array.shape[0]-1):
                    X=X+1
                    index_X=index[int(-1*X):]
                    percentageDroped=np.sum(array[index_X])/totalSaliency
        elif(percentageDroped>actualPercentage):
            X_=X-1
            index_X_=index[int(-1*X_):]
            percentageDroped_=np.sum(array[index_X_])/totalSaliency
            if(not (percentageDroped_<actualPercentage)):

                while(percentageDroped>actualPercentage and X>1):
                    X=X-1
                    index_X=index[int(-1*X):]
                    percentageDroped=np.sum(array[index_X])/totalSaliency

        indexes.append(index_X)
    return indexes





def generateNewSample(args):

    if(args.DataGenerationProcess==None):
        sample=np.random.normal(0,1,[args.NumTimeSteps,args.NumFeatures])

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
    return sample


def maskData(args,data,mask,noise=False):
    newData= np.zeros((data.shape))
    if(noise):
        noiseSample= generateNewSample(args)
        noiseSample=noiseSample.reshape(data.shape[1])
    for i in range(mask.shape[0]):
        newData[i,:]=data[i,:]
        cleanIndex = mask[i,:]
        cleanIndex=cleanIndex[np.logical_not(pd.isna(cleanIndex))]
        cleanIndex = cleanIndex.astype(np.int64)
        if(noise):
            newData[i,cleanIndex]=noiseSample[cleanIndex]
        else:
            newData[i,cleanIndex]=0

    return newData



def getRowColMaskIndex(mask,rows,columns):
    InColumn=np.zeros((mask.shape[0],columns),dtype=object)
    InRow=np.zeros((mask.shape[0],rows),dtype=object)
    InColumn[:,:]=False
    InRow[:,:]=False
    for i in range(mask.shape[0]):
        cleanIndex = mask[i,:]
        cleanIndex=cleanIndex[np.logical_not(pd.isna(cleanIndex))]
        cleanIndex = cleanIndex.astype(np.int64)
        for index in range(cleanIndex.shape[0]):
            InColumn[i,cleanIndex[index]%columns]=True
            InRow[i,int(cleanIndex[index]/columns)]=True
    return InRow,InColumn

                                         




################################## Get Accuracy Functions ##############################



def checkAccuracy(test_loader , model ,args,isCNN=False,returnLoss=False):
    
    model.eval()

    correct = 0
    total = 0
    if(returnLoss):
        loss=0
        criterion = nn.CrossEntropyLoss()

    for  (samples, labels)  in test_loader:
        if(isCNN):
            images = samples.reshape(-1, 1,args.NumTimeSteps, args.NumFeatures).to(device)
        else:
            images = samples.reshape(-1, args.NumTimeSteps, args.NumFeatures).to(device)

        outputs = model(images)
        if(returnLoss):
            labels = labels.to(device)
            loss+=criterion(outputs, labels).data
       
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()
    if(returnLoss):
        loss=loss/len(test_loader)
        return  (100 * float(correct) / total),loss

    return  (100 * float(correct) / total)



