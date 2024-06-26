#!/bin/bash

# Store parameters
SUBTASK="domainclassifier"
DATASET_NAME=""
LR=0.0005
DC_LOSS_GAIN=1.0
OUTPUT=".txt"


# Test on several Datasets
for dataset in {"pe_te_10percent_background","palm_mckellar_penguin_10percent_background","te_palm_10percent_background","te_mckellar_10percent_background"}
do
    MODEL_NAME=$"DAN_"$dataset
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $MODEL_NAME
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
done 


# Gridsearch DC loss gain
#MODEL_NAME_="pf_palm_dclossg_"
#for gain in {0.1,0.5,1.0,1.5,5,10}
#do
#    MODEL_NAME=$MODEL_NAME_$gain
    #MODEL_PATH=$"runs/detect/"$MODEL_NAME
   # OUTPUT_FILE=$MODEL_PATH$OUTPUT
  #  python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${gain}") >> $OUTPUT_FILE
 #   python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 


# Gridsearch LearningRate
#MODEL_NAME_="oldpfeifer_test_"
#for lr in {0.00001,0.00005,0.0005,0.0001,0.001,0.005,0.01}
#do
#    MODEL_NAME=$MODEL_NAME_$lr
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $OUTPUT_FILE
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${lr}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 