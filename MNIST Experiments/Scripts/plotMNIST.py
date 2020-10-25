import numpy as np
import matplotlib.pyplot as plt
import sys, os

import glob

graph_dir = '../Graphs/'




# [3, 10, 13]
# [2, 5, 14]
# [1, 35, 38]
# [18, 30, 32]
# [4, 6, 19]
# [8, 15, 23]
# [11, 21, 22]
# [0, 17, 26]
# [61, 84, 110]
# [7, 9, 12]

MNIST=['_0_index_4','_1_index_3','_2_index_2','_3_index_19','_4_index_5','_5_index_9','_6_index_12','_7_index_1','_8_index_62','_9_index_8']
# MNIST=['_0_index_11','_1_index_6','_2_index_36','_3_index_31','_4_index_7','_5_index_16','_6_index_22','_7_index_18','_8_index_85','_9_index_10']

methods = ["Grad","IG", "DL", "GS","DLS","SG","SVS","FA","FO"]



models=["LSTM","LSTMWithInputCellAttention","TCN","Transformer"]


Col_names_before=["Sample", "Grad","IG", "DL", "GS","DLS","SG","SVS","FA","OS"]

Col_names_after=["Sample", "TSR+\nGrad","TSR+\nIG", "TSR+\nDL", "TSR+\nGS","TSR+\nDLS","TSR+\nSG","TSR+\nSVS","TSR+\nFA","TSR+\nOS"]
  # _index3_digit1 0
  # _index2_digit2 1
  # _index1_digit3 2
  # _index18_digit4 3
  # _index4_digit5 4
  # _index8_digit6 5
  # _index11_digit7 6


models_name=["LSTM","LSTM+\nin.cell At.","TCN","Trans."]

# for modelName in models:
# 	row=len(MNIST)
# 	measurmentsCount = len(methods)+1
# 	# #figsize Wxh
# 	fig, (DS) = plt.subplots(row, measurmentsCount,sharex=True, figsize=(measurmentsCount,row))

# 	for b in range(len(MNIST)):
# 		DataName=MNIST[b]
# 		file =before_dir + "input_features"+MNIST[b]
# 		image = plt.imread(file+".png")
# 		DS[b,0].imshow(image)
# 		DS[b,0].axis('off')

# 	# LSTMWithInputCellAttention_IG_features28_index7_digit10

# 	for b in range(len(MNIST)):
# 		DataName=MNIST[b]
# 		for m ,  method in enumerate(methods):
# 			file =before_dir + modelName+ "_"+method+"_features28"+DataName
# 			image = plt.imread(file+".png")
# 			DS[b,m+1].imshow(image)
# 			DS[b,m+1].axis('off')
# 			if b==0:
# 				DS[b,m+1].set_title(Col_names[m+1], fontsize=12, fontweight='bold')




# 	plt.subplots_adjust(wspace=0, hspace=0)
# 	#(left, bottom, right, top)

# 	fig.tight_layout(rect=[0, 0,0.9,0.95])
# 	plt.savefig(results_dir+modelName+"_sup_before.png",  bbox_inches = 'tight',pad_inches = 0)
# 	plt.clf()


types=["before","after"]

for type in types:
	if(type=="after"):
		Col_names=Col_names_after
	else:
		Col_names=Col_names_before

	for  i , modelName in enumerate(models):
		row=len(MNIST)
		measurmentsCount = len(methods)+1
		# #figsize Wxh
		fig, (DS) = plt.subplots(row, measurmentsCount,sharex=True, figsize=(measurmentsCount,row))

		for b in range(len(MNIST)):
			DataName=MNIST[b]
			file =graph_dir + "Sample_MNIST"+MNIST[b]
			image = plt.imread(file+".png")
			DS[b,0].imshow(image)
			DS[b,0].axis('off')
		# else:
		# 	measurmentsCount = len(methods)
		# 	fig, (DS) = plt.subplots(row, measurmentsCount,sharex=True, figsize=(measurmentsCount,row))

		# LSTMWithInputCellAttention_IG_features28_index7_digit10
	# TCN_FO_MNIST_0_index_4
		for b in range(len(MNIST)):
			DataName=MNIST[b]
			for m ,  method in enumerate(methods):
				# LSTM_TSR_DLS_MNIST_0_index_4
				if(type =="after"):
					file =graph_dir + modelName+ "_TSR_"+method+"_MNIST"+DataName
				else:
					file =graph_dir + modelName+ "_"+method+"_MNIST"+DataName
				image = plt.imread(file+".png")

				DS[b,m+1].imshow(image)
				DS[b,m+1].axis('off')
				if b==0:
					DS[b,m+1].set_title(Col_names[m+1], fontsize=12, fontweight='bold')
				# else:
				# 	DS[b,m].imshow(image)
				# 	DS[b,m].axis('off')
				# 	if b==0:
				# 		DS[b,m].set_title(Col_names[m+1], fontsize=12, fontweight='bold')
				
					




		plt.subplots_adjust(wspace=0, hspace=0)
		#(left, bottom, right, top)

		fig.tight_layout(rect=[0, 0,0.9,0.95])
		plt.savefig(graph_dir+modelName+"_sup_"+type+".png",  bbox_inches = 'tight',pad_inches = 0)
		plt.clf()