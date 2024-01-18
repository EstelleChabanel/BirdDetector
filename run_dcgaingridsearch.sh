#!/bin/bash

# Store parameters
SUBTASK="domainclassifier"
DATASET_NAME="pe_palmyra_10percentbkgd"
LR=0.01
OUTPUT=".txt"

# Gridsearch DC loss gain
MODEL_NAME_=$"DAN_"$DATASET_NAME"_"
for gain in {0.1,0.5,1.0,1.5,5,10}
do
    MODEL_NAME=$MODEL_NAME_$gain
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${gain}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 



# Store parameters
SUBTASK="multidomainclassifier"
DATASET_NAME="pe_palmyra_10percentbkgd"
LR=0.001
OUTPUT=".txt"

# Gridsearch DC loss gain
MODEL_NAME_=$"multiDAN_"$DATASET_NAME"_"
for gain in {0.1,0.5,1.0,1.5,5,10}
do
    MODEL_NAME=$MODEL_NAME_$gain
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${gain}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 



# Store parameters
SUBTASK="detect"
DATASET_NAME="pe_palmyra_10percentbkgd"
LR=
OUTPUT=".txt"

# Gridsearch DC loss gain
MODEL_NAME_=$"YOLO_"$DATASET_NAME"_"
for gain in {0.1,0.5,1.0,1.5,5,10}
do
    MODEL_NAME=$MODEL_NAME_$gain
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${gain}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 
