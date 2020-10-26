from createSimulationDataProcesses import createSimulationDataProcesses












def createDatasets(args,DatasetsTypes,ImpTimeSteps,ImpFeatures,StartImpTimeSteps,StartImpFeatures,Loc1,Loc2,FreezeType,isMoving,isPositional,DataGenerationTypes):

	for i in range(len(DatasetsTypes)):

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
			





