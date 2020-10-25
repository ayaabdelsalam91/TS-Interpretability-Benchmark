import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
sys.path.append('../')
import Helper





def getSamples(args,fileName):


  Box_model= Helper.load_CSV(fileName)
  Saliency_rows=[]

  if(args.GradFlag):
      Saliency_rows.append(0)
  if(args.IGFlag):
      Saliency_rows.append(1)
  if(args.DLFlag):
      Saliency_rows.append(2)
  if(args.GSFlag):
      Saliency_rows.append(3)
  if(args.DLSFlag):
      Saliency_rows.append(4)
  if(args.SGFlag):
      Saliency_rows.append(5)
  if(args.ShapleySamplingFlag):
      Saliency_rows.append(6)
  if(args.FeaturePermutationFlag):
      Saliency_rows.append(7)
  if(args.FeatureAblationFlag):
      Saliency_rows.append(8)
  if(args.OcclusionFlag):
      Saliency_rows.append(9)
  Saliency_rows.append(10)


  Box_model=Box_model[Saliency_rows,1:]

  return  Box_model





index=[i for i in range(0,101,10)]



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



colors = [
          "red",
          "green",
          "cyan",
          "blue",
          "purple" , 
          "lime" , 
          "orange",
          "deeppink",
          "maroon",
          "pink",
          "brown",
          "black"]


DatasetsTypes= ["Middle", "RareTime","RareFeature"]

# DatasetsNames=[ 'Middle Box',]
DataGenerationTypes=[None]
DataGenerationNames=['Gaussian','Harmonic','Gaussian \n Process', 'Pseudo\nPeriodic','AR','Continuous\nAR','NARMA']
models=["LSTM" ,"LSTMWithInputCellAttention","TCN","Transformer"]

models_name=["LSTM","LSTM+\nInput-Cell Attention","TCN","Transformer"]


models=["LSTM" ,"LSTMWithInputCellAttention","TCN","Transformer"]

models_name=["LSTM","LSTM+\nin.cell At.","TCN","Transformer"]


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
        Saliency_Methods.append("SVS")
    if(args.FeaturePermutationFlag):
        Saliency_Methods.append("FP")
    if(args.FeatureAblationFlag):
        Saliency_Methods.append("FA")
    if(args.OcclusionFlag):
        Saliency_Methods.append("FO")
    Saliency_Methods.append("Random")

    row=len(DatasetsTypes)
    measurmentsCount=len(models)

    fig, (DS) = plt.subplots(row, measurmentsCount,sharex=True, figsize=(20,10))

    for x in range(len(DatasetsTypes)):


      #figsize Wxh

      print("Plotting", DatasetsTypes[x])
      
      args.DataName=DatasetsTypes[x]+"_Box"


      first_row_flag=True
      for m in range(len(models)):
          FileName=args.Masked_Acc_dir + args.DataName+"_"+models[m]+"_0_10_20_30_40_50_60_70_80_90_100_percentSal_rescaled.csv"
    


          data = getSamples(args,FileName)
          # print(data.shape)
          for i in range(data.shape[0]):
            if(i==len(Saliency_Methods)-1):
               DS[x,m].plot(index,data[i,:],color = colors[i],label=Saliency_Methods[i])
            else:

              DS[x,m].plot(index,data[i,:],color = colors[i],linestyle=':',label=Saliency_Methods[i])

            # DS[x,m].set_ylim([0, 100])
          # if(y==0):
          #   DS[y,m].set_title(models_name[m],fontsize=16)

          if(not first_row_flag):
              DS[x,m].tick_params(labelleft=False)  
          else:
            first_row_flag=False
          # if(m==len(models)-1):
          #     DS[x,m].set_ylabel(DataGenerationNames[y], fontsize=16)
          #     DS[x,m].yaxis.set_label_position("right")
          if(x!=len(DatasetsTypes)-1):
              DS[x,m].tick_params(labelbottom=False)

    handles, labels = DS[-1,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right',fontsize=16)

    # # fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0),
    # #           fancybox=True, shadow=True, ncol=len(labels), fontsize=25)
    # #(left, bottom, right, top)

    fig.tight_layout(rect=[0.11, 0.15,0.85,0.95])

    fig.text(0.5, 0.11, '% of features masked', ha='center',fontsize=16)
    fig.text(0.09, 0.5, 'Model Accuracy', va='center', rotation='vertical',fontsize=16)
    # fig.suptitle(DatasetsNames[x],fontsize=22)

    plt.savefig(args.Graph_dir+DatasetsTypes[x]+'_AccuracyDrop_rescaled_pres2.png',  bbox_inches = 'tight',pad_inches = 0)


   




def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--NumTimeSteps',type=int,default=50)
    parser.add_argument('--NumFeatures',type=int,default=50)
    parser.add_argument('--DataGenerationProcess', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default="../../Datasets/")
    parser.add_argument('--Mask_dir', type=str, default='../../Results/Saliency_Masks/')
    parser.add_argument('--Masked_Acc_dir', type=str, default= "../../Results/MaskedAccuracy/")

    parser.add_argument('--Saliency_Distribution_dir', type=str, default= "../../Results/Saliency_Distribution/")
    parser.add_argument('--Saliency_dir', type=str, default='../../Results/Saliency_Values/')



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


    parser.add_argument('--Graph_dir', type=str, default='../../Graphs/AccuracyDrop/')

    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
plt.clf()