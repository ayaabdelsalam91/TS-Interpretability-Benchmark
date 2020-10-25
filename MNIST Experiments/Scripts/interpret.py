import torch
import torch.optim as optim
import torch.nn.functional as F
import sys, os
sys.path.append("../../Scripts")
from model import Transformer,TCN,LSTMWithInputCellAttention,LSTM
import numpy as np
import argparse
import random
from utils import data_generator
import Helper
from Plotting import plotExampleBox
from torch.autograd import Variable
from  sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    Saliency,
    NoiseTunnel,
    ShapleyValueSampling,
    FeaturePermutation,
    FeatureAblation,
    Occlusion

)




def getTwoStepRescaling(Grad,input, sequence_length,input_size, TestingLabel,hasBaseline=None,hasFeatureMask=None,hasSliding_window_shapes=None):
    assignment=input[0,0,0]
    timeGrad=np.zeros((1,sequence_length))
    inputGrad=np.zeros((input_size,1))
    newGrad=np.zeros((input_size, sequence_length))
    if(hasBaseline==None):  
        ActualGrad = Grad.attribute(input,target=TestingLabel).data.cpu().numpy()
    else:
        if(hasFeatureMask!=None):
            ActualGrad = Grad.attribute(input,baselines=hasBaseline, target=TestingLabel,feature_mask=hasFeatureMask).data.cpu().numpy()    
        elif(hasSliding_window_shapes!=None):
            ActualGrad = Grad.attribute(input,sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
        else:
            ActualGrad = Grad.attribute(input,baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()


    for t in range(sequence_length):
        timeGrad[:,t] = np.mean(np.absolute(ActualGrad[0,:,t]))


    # for t in range(sequence_length):
    #     newInput = input.clone()
    #     newInput[:,:,t]=assignment

    #     if(hasBaseline==None):  
    #         timeGrad_perTime = Grad.attribute(newInput,target=TestingLabel).data.cpu().numpy()
    #     else:
    #         if(hasFeatureMask!=None):
    #             timeGrad_perTime = Grad.attribute(newInput,baselines=hasBaseline, target=TestingLabel,feature_mask=hasFeatureMask).data.cpu().numpy()    
    #         elif(hasSliding_window_shapes!=None):
    #             timeGrad_perTime = Grad.attribute(newInput,sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
    #         else:
    #             timeGrad_perTime = Grad.attribute(newInput,baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()


    #     timeGrad_perTime= np.absolute(ActualGrad - timeGrad_perTime)
    #     timeGrad[:,t] = np.sum(timeGrad_perTime)



    timeContibution=preprocessing.minmax_scale(timeGrad, axis=1)
    meanTime = np.quantile(timeContibution, .55)    

    
    for t in range(sequence_length):
        if(timeContibution[0,t]>meanTime):
            for c in range(input_size):
                newInput = input.clone()
                newInput[:,c,t]=assignment

                if(hasBaseline==None):  
                    inputGrad_perInput = Grad.attribute(newInput,target=TestingLabel).data.cpu().numpy()
                else:
                    if(hasFeatureMask!=None):
                        inputGrad_perInput = Grad.attribute(newInput,baselines=hasBaseline, target=TestingLabel,feature_mask=hasFeatureMask).data.cpu().numpy()    
                    elif(hasSliding_window_shapes!=None):
                        inputGrad_perInput = Grad.attribute(newInput,sliding_window_shapes=hasSliding_window_shapes, baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()
                    else:
                        inputGrad_perInput = Grad.attribute(newInput,baselines=hasBaseline, target=TestingLabel).data.cpu().numpy()



                inputGrad_perInput=np.absolute(ActualGrad - inputGrad_perInput)
                inputGrad[c,:] = np.sum(inputGrad_perInput)
                # print(t,c,np.sum(inputGrad_perInput),np.sum(input.data.cpu().numpy()))
            # featureContibution=inputGrad
            featureContibution= preprocessing.minmax_scale(inputGrad, axis=0)
        else:
            featureContibution=np.ones((input_size,1))*0.1
       
        # meanFeature=np.mean(featureContibution, axis=0)
        # for c in range(input_size): 
        #     if(featureContibution[c,0]<=meanFeature):
        #         featureContibution[c,0]=0
        for c in range(input_size):

           newGrad [c,t]= timeContibution[0,t]*featureContibution[c,0]
           # if(newGrad [c,t]==0):
           #  print(timeContibution[0,t],featureContibution[c,0])
    return newGrad


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models=["LSTMWithInputCellAttention","TCN","Transformer"]
def main(args):

    train_loader, test_loader = data_generator(args.data_dir,1)

    for m in range(len(models)):

        model_name = "model_{}_NumFeatures_{}".format(models[m],args.NumFeatures)
        model_filename = args.model_dir + 'm_' + model_name + '.pt'
        pretrained_model = torch.load(open(model_filename, "rb"),map_location=device) 
        pretrained_model.to(device)



        if(args.GradFlag):
            Grad = Saliency(pretrained_model)
        if(args.IGFlag):
            IG = IntegratedGradients(pretrained_model)
        if(args.DLFlag):
            DL = DeepLift(pretrained_model)
        if(args.GSFlag):
            GS = GradientShap(pretrained_model)
        if(args.DLSFlag):
            DLS = DeepLiftShap(pretrained_model)                 
        if(args.SGFlag):
            Grad_ = Saliency(pretrained_model)
            SG = NoiseTunnel(Grad_)
        if(args.ShapleySamplingFlag):
            SS = ShapleyValueSampling(pretrained_model)
        if(args.GSFlag):
            FP = FeaturePermutation(pretrained_model)
        if(args.FeatureAblationFlag):
            FA = FeatureAblation(pretrained_model)         
        if(args.OcclusionFlag):
            OS = Occlusion(pretrained_model)

        timeMask=np.zeros((args.NumTimeSteps, args.NumFeatures),dtype=int)
        featureMask=np.zeros((args.NumTimeSteps, args.NumFeatures),dtype=int)
        for i in  range (args.NumTimeSteps):
            timeMask[i,:]=i

        for i in  range (args.NumTimeSteps):
            featureMask[:,i]=i

        indexes = [[] for i in range(5,10)]
        for i ,(data, target) in enumerate(test_loader):
            if(target==5 or target==6 or target==7 or target==8 or target==9):
                index=target-5

                if(len(indexes[index])<1):
                    indexes[index].append(i)
        for j, index in enumerate(indexes):
            print(index)
        # indexes = [[21],[17],[84],[9]]

        for j, index in enumerate(indexes):
            print("Getting Saliency for number", j+1)
            for i, (data, target) in enumerate(test_loader):
                if(i in index):
                        
                    labels =  target.to(device)
             
                    input = data.reshape(-1, args.NumTimeSteps, args.NumFeatures).to(device)
                    input = Variable(input,  volatile=False, requires_grad=True)

                    baseline_single=torch.Tensor(np.random.random(input.shape)).to(device)
                    baseline_multiple=torch.Tensor(np.random.random((input.shape[0]*5,input.shape[1],input.shape[2]))).to(device)
                    inputMask= np.zeros((input.shape))
                    inputMask[:,:,:]=timeMask
                    inputMask =torch.Tensor(inputMask).to(device)
                    mask_single= torch.Tensor(timeMask).to(device)
                    mask_single=mask_single.reshape(1,args.NumTimeSteps, args.NumFeatures).to(device)

                    Data=data.reshape(args.NumTimeSteps, args.NumFeatures).data.cpu().numpy()
                    
                    target_=int(target.data.cpu().numpy()[0])

                    plotExampleBox(Data,args.Graph_dir+'Sample_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)




                    if(args.GradFlag):
                        attributions = Grad.attribute(input, \
                                                      target=labels)
                        
                        saliency_=Helper.givenAttGetRescaledSaliency(args,attributions)

                        plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_Grad_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)
                        if(args.TSRFlag):
                            TSR_attributions =  getTwoStepRescaling(Grad,input, args.NumFeatures,args.NumTimeSteps, labels,hasBaseline=None)
                            TSR_saliency=Helper.givenAttGetRescaledSaliency(args,TSR_attributions,isTensor=False)
                            plotExampleBox(TSR_saliency,args.Graph_dir+models[m]+'_TSR_Grad_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)



                    if(args.IGFlag):
                        attributions = IG.attribute(input,  \
                                                    baselines=baseline_single, \
                                                    target=labels)
                        saliency_=Helper.givenAttGetRescaledSaliency(args,attributions)

                        plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_IG_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)
                        if(args.TSRFlag):
                            TSR_attributions =  getTwoStepRescaling(IG,input, args.NumFeatures,args.NumTimeSteps, labels,hasBaseline=baseline_single)
                            TSR_saliency=Helper.givenAttGetRescaledSaliency(args,TSR_attributions,isTensor=False)
                            plotExampleBox(TSR_saliency,args.Graph_dir+models[m]+'_TSR_IG_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)




                    if(args.DLFlag):
                        attributions = DL.attribute(input,  \
                                                    baselines=baseline_single, \
                                                    target=labels)
                        saliency_=Helper.givenAttGetRescaledSaliency(args,attributions)
                        plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_DL_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)


                        if(args.TSRFlag):
                            TSR_attributions =  getTwoStepRescaling(DL,input, args.NumFeatures,args.NumTimeSteps, labels,hasBaseline=baseline_single)
                            TSR_saliency=Helper.givenAttGetRescaledSaliency(args,TSR_attributions,isTensor=False)
                            plotExampleBox(TSR_saliency,args.Graph_dir+models[m]+'_TSR_DL_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)




                    if(args.GSFlag):

                        attributions = GS.attribute(input,  \
                                                    baselines=baseline_multiple, \
                                                    stdevs=0.09,\
                                                    target=labels)
                        saliency_=Helper.givenAttGetRescaledSaliency(args,attributions)
                        plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_GS_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)

 
                        if(args.TSRFlag):
                            TSR_attributions =  getTwoStepRescaling(GS,input, args.NumFeatures,args.NumTimeSteps, labels,hasBaseline=baseline_multiple)
                            TSR_saliency=Helper.givenAttGetRescaledSaliency(args,TSR_attributions,isTensor=False)
                            plotExampleBox(TSR_saliency,args.Graph_dir+models[m]+'_TSR_GS_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)


                    if(args.DLSFlag):

                        attributions = DLS.attribute(input,  \
                                                    baselines=baseline_multiple, \
                                                    target=labels)
                        saliency_=Helper.givenAttGetRescaledSaliency(args,attributions)
                        plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_DLS_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)
                        if(args.TSRFlag):
                            TSR_attributions =  getTwoStepRescaling(DLS,input, args.NumFeatures,args.NumTimeSteps, labels,hasBaseline=baseline_multiple)
                            TSR_saliency=Helper.givenAttGetRescaledSaliency(args,TSR_attributions,isTensor=False)
                            plotExampleBox(TSR_saliency,args.Graph_dir+models[m]+'_TSR_DLS_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)



                    if(args.SGFlag):
                        attributions = SG.attribute(input, \
                                                    target=labels)
                        saliency_=Helper.givenAttGetRescaledSaliency(args,attributions)
                        plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_SG_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)
                        if(args.TSRFlag):
                            TSR_attributions =  getTwoStepRescaling(SG,input, args.NumFeatures,args.NumTimeSteps, labels)
                            TSR_saliency=Helper.givenAttGetRescaledSaliency(args,TSR_attributions,isTensor=False)
                            plotExampleBox(TSR_saliency,args.Graph_dir+models[m]+'_TSR_SG_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)


                    if(args.ShapleySamplingFlag):
                        attributions = SS.attribute(input, \
                                        baselines=baseline_single, \
                                        target=labels,\
                                        feature_mask=inputMask)
                        saliency_=Helper.givenAttGetRescaledSaliency(args,attributions)
                        plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_SVS_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)
                        if(args.TSRFlag):
                            TSR_attributions =  getTwoStepRescaling(SS,input, args.NumFeatures,args.NumTimeSteps, labels,hasBaseline=baseline_single,hasFeatureMask=inputMask)
                            TSR_saliency=Helper.givenAttGetRescaledSaliency(args,TSR_attributions,isTensor=False)
                            plotExampleBox(TSR_saliency,args.Graph_dir+models[m]+'_TSR_SVS_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)
                    # if(args.FeaturePermutationFlag):
                    #     attributions = FP.attribute(input, \
                    #                     target=labels),
                    #                     # perturbations_per_eval= 1,\
                    #                     # feature_mask=mask_single)
                    #     saliency_=Helper.givenAttGetRescaledSaliency(args,attributions) 
                    #     plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_FP',greyScale=True)


                    if(args.FeatureAblationFlag):
                        attributions = FA.attribute(input, \
                                        target=labels)
                                        # perturbations_per_eval= input.shape[0],\
                                        # feature_mask=mask_single)
                        saliency_=Helper.givenAttGetRescaledSaliency(args,attributions) 
                        plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_FA_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)
                        if(args.TSRFlag):
                            TSR_attributions =  getTwoStepRescaling(FA,input, args.NumFeatures,args.NumTimeSteps, labels)
                            TSR_saliency=Helper.givenAttGetRescaledSaliency(args,TSR_attributions,isTensor=False)
                            plotExampleBox(TSR_saliency,args.Graph_dir+models[m]+'_TSR_FA_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)

                    if(args.OcclusionFlag):
                        attributions = OS.attribute(input, \
                                        sliding_window_shapes=(1,int(args.NumFeatures/10)),
                                        target=labels,
                                        baselines=baseline_single)
                        saliency_=Helper.givenAttGetRescaledSaliency(args,attributions) 

                        plotExampleBox(saliency_[0],args.Graph_dir+models[m]+'_FO_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)
                        if(args.TSRFlag):
                            TSR_attributions =  getTwoStepRescaling(OS,input, args.NumFeatures,args.NumTimeSteps, labels,hasBaseline=baseline_single,hasSliding_window_shapes= (1,int(args.NumFeatures/10)))
                            TSR_saliency=Helper.givenAttGetRescaledSaliency(args,TSR_attributions,isTensor=False)
                            plotExampleBox(TSR_saliency,args.Graph_dir+models[m]+'_TSR_FO_MNIST_'+str(target_)+'_index_'+str(i+1),greyScale=True)






def parse_arguments(argv):
    parser = argparse.ArgumentParser()


    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 64)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='dropout applied to layers (default: 0.05)')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit (default: 20)')
    parser.add_argument('--ksize', type=int, default=7,
                        help='kernel size (default: 7)')


    parser.add_argument('--TSRFlag', type=bool, default=False)
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

    parser.add_argument('--Graph_dir', type=str, default='../Graphs/')






    parser.add_argument('--levels', type=int, default=8,
                        help='# of levels (default: 8)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval (default: 100')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='initial learning rate (default: 2e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--nhid', type=int, default=25,
                        help='number of hidden units per layer (default: 25)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')

    parser.add_argument('--n_classes', type=int, default=10)


    parser.add_argument('--data_dir', type=str, default="../Data/")
    parser.add_argument('--model_dir', type=str, default="../Models/")


    parser.add_argument('--NumTimeSteps',type=int,default=28)
    parser.add_argument('--NumFeatures',type=int,default=28)

    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=4)

    parser.add_argument('--attention_hops', type=int, default=28)
    parser.add_argument('--d_a', type=int, default=30)

    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
