#!/bin/bash

# Store parameters
MODEL_NAME_="pe_palm_dclossg_"
SUBTASK="domainclassifier"
DATASET_NAME="pe_palmyra_10percentbkgd"
DATASETS={"global_birds_penguins","global_birds_palmyra"}
LR=0.001
DC_LOSS_GAIN=1.0
DC_LOSS_GAINS=(0.1, 0.5, 1.0, 1.5, 5, 10)
OUTPUT=".txt"

# Gridsearch DC loss gain
#for gain in {0.1,0.5,1.0,1.5,5,10}
#do
#    MODEL_NAME=$MODEL_NAME_$gain
#    OUTPUT_FILE=$MODEL_NAME$OUTPUT
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --dcloss-gain $(bc -l <<<"${gain}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 

# Gridsearch LearningRate
for lr in {0.0005,0.0001,0.001,0.005,0.01}
do
    MODEL_NAME=$MODEL_NAME_$lr
    OUTPUT_FILE=$MODEL_NAME$OUTPUT
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 