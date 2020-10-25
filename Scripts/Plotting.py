import matplotlib.pyplot as plt
import timesynth as ts
import numpy as np





def plotExampleBox(input, saveLocation, show=False,greyScale=False,flip=False):
    
    if(flip):
        input=np.transpose(input)
    fig, ax = plt.subplots()

    if(greyScale):
        cmap='gray'
    else:
        cmap='seismic'
    
    plt.axis('off')
  
    cax = ax.imshow(input, interpolation='nearest', cmap=cmap)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(saveLocation+'.png' , bbox_inches = 'tight',pad_inches = 0)

    if(show):
        plt.show()
    plt.close()



def plotExampleProcesses(sample,saveLocation,show=False,color='r'):

	times = [i for i  in range (sample.shape[0])]
	for j in range(sample.shape[1]):
		plt.plot(times,sample[:,j],color =color)

	# plt.gca().set_axis_off()
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
	plt.margins(0,0)
	# plt.gca().xaxis.set_major_locator(plt.NullLocator())
	# plt.gca().yaxis.set_major_locator(plt.NullLocator())

	plt.savefig(saveLocation+'.png' , bbox_inches = 'tight',pad_inches = 0)

	if(show):
		plt.show()
	plt.close()