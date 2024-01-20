#!/bin/bash

# Store parameters
SUBTASK="domainclassifier"
DATASET_NAME="pe_palmyra_10percentbkgd"
DC_LOSS_GAIN=1.0
OUTPUT=".txt"

# Gridsearch LearningRate
#MODEL_NAME_=$"DAN_"$DATASET_NAME$"_"
#for lr in {0.0005,0.001,0.005,0.01,0.05,0.1}    #{0.00005,0.0001,0.0005,0.001,0.005,0.01}
#do
#    MODEL_NAME=$MODEL_NAME_$lr
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $OUTPUT_FILE
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${lr}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 



# Store parameters
#SUBTASK="multidomainclassifier"
#DATASET_NAME="pe_palmyra_10percentbkgd"
#DC_LOSS_GAIN=1.0
#OUTPUT=".txt"

# Gridsearch LearningRate
#MODEL_NAME_=$"multiDAN_"$DATASET_NAME"_"
#for lr in {0.00005,0.0001,0.0005,0.001,0.005,0.01}
#do
#    MODEL_NAME=$MODEL_NAME_$lr
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $OUTPUT_FILE
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${lr}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 


# Store parameters
#SUBTASK="featuresdistance"
#DATASET_NAME="pe_palmyra_10percentbkgd"
#DC_LOSS_GAIN=1.0
#OUTPUT=".txt"

## Gridsearch LearningRate
#MODEL_NAME_=$"FeatDist_"$DATASET_NAME"_"
#for lr in {0.0001,0.0005,0.001,0.005,0.01,0.05,0.1}  #{0.00005,0.0001,0.0005,0.001,0.005,0.01}
#do
 #   MODEL_NAME=$MODEL_NAME_$lr
 #   MODEL_PATH=$"runs/detect/"$MODEL_NAME
 #   OUTPUT_FILE=$MODEL_PATH$OUTPUT
 #   echo $OUTPUT_FILE
 #   python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${lr}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
 #   python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 


# Store parameters
SUBTASK="detect"
DATASET_NAME="pe_palmyra_10percentbkgd"
DC_LOSS_GAIN=1.0
OUTPUT=".txt"

# Gridsearch LearningRate
MODEL_NAME_=$"YOLO_"$DATASET_NAME"_"
for lr in {0.01,0.05,0.1}  # 0.0001,0.0005,0.001,0.005, {0.00005,0.0001,0.0005,0.001,0.005,0.01}
do
    MODEL_NAME=$MODEL_NAME_$lr
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $OUTPUT_FILE
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${lr}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 