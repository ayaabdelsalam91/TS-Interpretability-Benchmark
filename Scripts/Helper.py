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




def reOrderLabels(Labels):
	uniqueLabels =  list(set(Labels))
	outLabels=[]
	for label in Labels:
		outLabels.append(uniqueLabels.index(label))
	return outLabels


def getIndexOfXhighestFeatures(array,X):
    return np.argpartition(array, int(-1*X))[int(-1*X):]


def getAverageOfMaxX(array,X):
    index = getIndexOfXhighestFeatures(array,X)
    avg=np.mean(array[index])
    return avg
def getIndexOfXhighestSalientValues(array,percentage):
    totalSaliency=np.sum(array)
    X=1
    index=getIndexOfXhighestFeatures(array,X)
    # print(index)
    percentageDroped=np.sum(array[index])/totalSaliency
    if(percentageDroped<percentage):
        while(percentageDroped<percentage and X<array.shape[0]-1):
            X=X+1
            index=getIndexOfXhighestFeatures(array,X)
            percentageDroped=np.sum(array[index])/totalSaliency

    elif(percentageDroped>percentage):
        while(percentageDroped>percentage and X>1):

            X=X-1
            index=getIndexOfXhighestFeatures(array,X)
            percentageDroped=np.sum(array[index])/totalSaliency
    return index

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

def getIndexOfXlowestFeatures(array,X):
    return np.argpartition(array, X)[:X]


def rescale(image,scale=None):
    if(scale==None):
        new= minmax_scale(image)
    else:
        new= minmax_scale(image,scale)
    return new


def getMaxPerc(Acc,AccList,Percentages):
    Orignal=AccList[0]
    for i in range(1,len(AccList)):

        Drop=round(Orignal,2)-round(AccList[i],2)
        if(Drop>Acc):
            return Percentages[i-1]
    return Percentages[-1]

def prepareMaskFile(mask,complement=False):
    if(not complement):
        newMask= np.zeros((mask.shape))
        for i in range(mask.shape[0]):
            cleanIndex = mask[i,:]
            cleanIndex=cleanIndex[np.logical_not(np.isnan(cleanIndex))]
            cleanIndex = cleanIndex.astype(np.int64)
            newMask[i,cleanIndex]=1
    else:
        newMask= np.ones((mask.shape))
        for i in range(mask.shape[0]):
            cleanIndex = mask[i,:]
            cleanIndex=cleanIndex[np.logical_not(np.isnan(cleanIndex))]
            cleanIndex = cleanIndex.astype(np.int64)
            newMask[i,cleanIndex]=0
    return newMask


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

def mask_data(data,mask):
    result = data* (1. - mask)
    
    return result



def getIndexInfo(index,informativeIndex , width):
    row    = (int)(index / width)
    column = index % width
    isInformative = index in informativeIndex

    return row , column, isInformative

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





def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)       



################################## Adversial Attack ##############################






def multiple_mini_batch_attack(adversary, loader,args,device="cuda", norm=None, num_batch=None,mask=None,save=False,saveDataSize=None):
    lst_label = []
    lst_pred = []
    lst_advpred = []
    lst_dist = []

    _norm_convert_dict = {"Linf": "inf", "L2": 2, "L1": 1}
    if norm in _norm_convert_dict:
        norm = _norm_convert_dict[norm]

    if norm == "inf":
        def dist_func(x, y):
            return (x - y).view(x.size(0), -1).max(dim=1)[0]
    elif norm == 1 or norm == 2:
        from advertorch.utils import _get_norm_batch

        def dist_func(x, y):
            return _get_norm_batch(x - y, norm)
    else:
        assert norm is None


    idx_batch = 0
    if(save):
        advData=np.zeros(saveDataSize)
        saveIndex=0
    for  (samples, labels,mask)  in loader:
        samples = samples.reshape(-1, args.sequence_length, args.input_size).to(device)
        labels =labels.to(device)
        if(mask is not None):
            newMask=mask.reshape(-1, args.sequence_length, args.input_size).to(device)
            adv = adversary.perturb(samples, labels,newMask)
        else:
            adv = adversary.perturb(samples, labels)
        if(save):
            perturbedData = adv.detach().cpu().numpy()
            advData[saveIndex:saveIndex+args.batch_size,:,:] =perturbedData
            saveIndex+=args.batch_size
        advpred = predict_from_logits(adversary.predict(adv))
        pred = predict_from_logits(adversary.predict(samples))
        lst_label.append(labels)
        lst_pred.append(pred)
        lst_advpred.append(advpred)
        if norm is not None:
            lst_dist.append(dist_func(samples, adv))

        idx_batch += 1
        if idx_batch == num_batch:
            break

    if(save):
        return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred), \
        torch.cat(lst_dist) if norm is not None else None,advData
    else:
        return torch.cat(lst_label), torch.cat(lst_pred), torch.cat(lst_advpred), \
            torch.cat(lst_dist) if norm is not None else None



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



