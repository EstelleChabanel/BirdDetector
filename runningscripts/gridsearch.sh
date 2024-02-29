#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5120

salloc --time=6:00:00 --partition=gpu_devel --gpus=1 --ntasks=1 --cpus-per-task=10 --mem-per-cpu=5120

module load CUDA
module load cuDNN

source activate pdm


# Store parameters
SUBTASK="unsuperviseddomainclassifier"
DATASET_NAME="poland_mckellar_unsupervised"

# Gridsearch DC loss gain
MODEL_NAME_=$"UNsupDAN_"$DATASET_NAME"_"
for gain in {0.5,1.0} #{0.25,0.5,0.75,1.0,1.5,2,3,5,10}
do
    MODEL_NAME=$MODEL_NAME_$"dcgain"$gain
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"  #--dcloss-gain $(bc -l <<<"${$"1.5"}") 
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 



# Store parameters
SUBTASK="unsupervisedmultidomainclassifier" #"featdist"
DATASET_NAME="poland_mckellar_unsupervised"   #"pe_10percent_background_unsupervised_moretarget"
LR=0.01
OUTPUT=".txt"


MODEL_NAME_=$"UNsupMultiDAN_"$DATASET_NAME"_"
for gain1 in {"1.0","0.5","1.5"}  
do
    for gain2 in {"1.0","0.5","1.5"}
    do 
        for gain3 in {"1.0","0.5","1.5"}
            MODEL_NAME=$MODEL_NAME_$"dcgain_"$gain1$"_"$gain2$"_"$gain3
            MODEL_PATH=$"runs/detect/"$MODEL_NAME
            OUTPUT_FILE=$MODEL_PATH$OUTPUT
            python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --gains $gain1,$gain2,$gain3 --default-param True
            python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
        done
    done
done
#





# Store parameters
SUBTASK="unsupervisedfeaturesdistance"
DATASET_NAME="poland_mckellar_unsupervised"
#LR=0.01
OUTPUT=".txt"

# Gridsearch DC loss gain
MODEL_NAME_=$"UNsupFeatDist_"$DATASET_NAME"_"
for gain in {0.25,0.5,1.0,1.5} #{0.25,0.5,0.75,1.0,1.5,2,3,5,10}
do
    MODEL_NAME=$MODEL_NAME_$"dcgain"$gain
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
 #   OUTPUT_FILE=$MODEL_PATH$OUTPUT
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --dcloss-gain $(bc -l <<<"${gain}") 
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 
