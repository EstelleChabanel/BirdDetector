#!/bin/bash

# Store parameters
#SUBTASK="domainclassifier"
#DATASET_NAME="pe_palmyra_10percentbkgd"
#LR=0.01
#OUTPUT=".txt"

## Gridsearch DC loss gain
#MODEL_NAME_=$"DAN_"$DATASET_NAME"_"
#for gain in {3,7}  #0.1,0.5,   0.5,0.75,2,3
#do
#    MODEL_NAME=$MODEL_NAME_$gain
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${gain}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 



# Store parameters
#SUBTASK="multidomainclassifier"
#DATASET_NAME="pe_palmyra_10percentbkgd"
#LR=0.01
#OUTPUT=".txt"

# Gridsearch DC loss gain
#MODEL_NAME_=$"multiDAN_"$DATASET_NAME"_"
#for gain in {5,10} #{0.25,0.5,0.75,1.0,1.5,2,3,5,10}
#do
#    MODEL_NAME=$MODEL_NAME_$gain
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${gain}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 


# Store parameters
SUBTASK="featuresdistance"
DATASET_NAME="pe_palmyra_10percentbkgd"
LR=0.01
OUTPUT=".txt"

# Gridsearch DC loss gain
MODEL_NAME_=$"featdist_"$DATASET_NAME"_"
for gain in {3,5,10}  # 0.5,0.75,1.0,1.5,2,3,5,10   0.75,1.0,1.5,
do
    MODEL_NAME=$MODEL_NAME_$gain
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${gain}") >> $OUTPUT_FILE
    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 

