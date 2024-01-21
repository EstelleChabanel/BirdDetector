#!/bin/bash

# Store parameters
SUBTASK="domainclassifier"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=1.0
OUTPUT=".txt"


# Test on several Datasets
for dataset in {"pe_pf_palm_10percent_background","pe_te_10percent_background","pe_pf_te_10percent_background","te_palm_10percent_background"}
do
    MODEL_NAME=$"DAN_"$dataset
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $MODEL_NAME
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
done 



# Store parameters
SUBTASK="multidomainclassifier"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=1.0
OUTPUT=".txt"


# Test on several Datasets
for dataset in {"pe_pf_palm_10percent_background","pe_te_10percent_background","pe_pf_te_10percent_background","te_palm_10percent_background"}
do
    MODEL_NAME=$"multiDAN_"$dataset
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $MODEL_NAME
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
done 


# Store parameters
SUBTASK="featsdist"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=1.0
OUTPUT=".txt"


# Test on several Datasets
for dataset in {"pe_pf_palm_10percent_background","pe_te_10percent_background","pe_pf_te_10percent_background","te_palm_10percent_background"}
do
    MODEL_NAME=$"featdist_"$dataset
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $MODEL_NAME
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
done 



# Store parameters
SUBTASK="detect"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=1.0
OUTPUT=".txt"


# Test on several Datasets
for dataset in {"pe_pf_palm_10percent_background","pe_te_10percent_background","pe_pf_te_10percent_background","te_palm_10percent_background"}
do
    MODEL_NAME=$"YOLO_"$dataset
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $MODEL_NAME
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
done 