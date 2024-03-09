# sed -i 's/\r//' copy_data.sh

for n in subject10;
do
    # mkdir -p "/mnt/c/Users/antoi/Documents/MyRepositories/opencap-processing-fork/Data/Benchmark/$n/OpenSimData/Dynamics"
    # sshpass -p 'Stanford!' scp -r clarkadmin@171.65.102.146:Documents/MyRepositories_Antoine/opencap-processing-fork/opencap-processing/Data/Benchmark/$n/OpenSimData/Dynamics/* "/mnt/c/Users/antoi/Documents/MyRepositories/opencap-processing-fork/Data/Benchmark/$n/OpenSimData/Dynamics"
    # sshpass -p 'Stanford!' scp -r clarkadmin@171.65.102.146:Documents/MyRepositories_Antoine/opencap-processing-fork/opencap-processing/Data/Benchmark/$n/OpenSimData/Model/* "/mnt/c/Users/antoi/Documents/MyRepositories/opencap-processing-fork/Data/Benchmark/$n/OpenSimData/Model"  
    sshpass -p 'Stanford!' rsync -avz --ignore-existing clarkadmin@171.65.102.146:Documents/MyRepositories_Antoine/opencap-processing-fork/opencap-processing/Data/Benchmark_updated/$n/ "/mnt/c/Users/antoi/Documents/MyRepositories/opencap-processing-fork/Data/Benchmark_updated/$n"
    # sshpass -p 'Stanford!' rsync -avz --ignore-existing "/mnt/c/Users/antoi/Documents/MyRepositories/opencap-processing-fork/Data/Benchmark_updated/$n/" clarkadmin@171.65.102.146:Documents/MyRepositories_Antoine/opencap-processing-fork/opencap-processing/Data/Benchmark_updated/$n 
    
    




	# if [ ! -d "clarkadmin@171.65.102.146:Documents/MyRepositories_Antoine/opencap-processing-fork/opencap-processing/Data/Benchmark/$n/sessionMetadata.yaml" ]; then

    #     sshpass -p 'Stanford!' mkdir -p clarkadmin@171.65.102.146:Documents/MyRepositories_Antoine/opencap-processing-fork/opencap-processing/Data/Benchmark/$n
    # 	sshpass -p 'Stanford!' scp -r "/mnt/c/Users/antoi/Documents/MyRepositories/opencap-processing-fork/Data/Benchmark/$n/*" clarkadmin@171.65.102.146:Documents/MyRepositories_Antoine/opencap-processing-fork/opencap-processing/Data/Benchmark/$n
	# fi	
done;