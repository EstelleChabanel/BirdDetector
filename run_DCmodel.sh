#!/bin/bash

# Store parameters
SUBTASK="domainclassifier"
DATASET_NAME="pe_palmyra_10percentbkgd"
#DATASETS={"global_birds_penguins","global_birds_palmyra"}
LR=0.0005
DC_LOSS_GAIN=1.0
DC_LOSS_GAINS=(0.1, 0.5, 1.0, 1.5, 5, 10)
OUTPUT=".txt"

# Gridsearch DC loss gain
MODEL_NAME_="pe_palm_dclossg_"
for gain in {10,12} #{0.1,0.5,1.0,1.5,5,10}
do
    MODEL_NAME=$MODEL_NAME_$gain
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${gain}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 

# Gridsearch LearningRate
#MODEL_NAME_="pe_palm_lr_"
#for lr in {0.00001,0.00005} #{0.0005,0.0001,0.001,0.005,0.01}
#do
#    MODEL_NAME=$MODEL_NAME_$lr
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $OUTPUT_FILE
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${lr}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 