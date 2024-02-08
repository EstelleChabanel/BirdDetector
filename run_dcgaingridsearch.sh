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
SUBTASK="multidomainclassifier" #"featdist"
DATASET_NAME="pe_palmyra_10percentbkgd"
LR=0.01
OUTPUT=".txt"

# Gridsearch DC loss gain
#MODEL_NAME_=$"multiDAN_"$DATASET_NAME"_"
#for gain1 in {"0.25","1.0","1.5","0.75","1.25","1.75","0.25","2"}  
#do
    #for gain2 in {"0.5","1.0","1.5"}   # "1.25","0.25","1.75","2","0.75",
    #do
        #for gain3 in {"1.0","1.5"}   # ,"1.25","1.75","2","0.75","0.25"  #{"0.25","0.5","0.75","1.0","1.25","1.5","1.75","2"}
        #do
            #MODEL_NAME=$MODEL_NAME_$"dcgain"$gain1$"_"$gain2$"_"$gain3
            #MODEL_PATH=$"runs/detect/"$MODEL_NAME
            #OUTPUT_FILE=$MODEL_PATH$OUTPUT
            #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --gains $gain1,$gain2,$gain3 --default-param True #>> $OUTPUT_FILE
            #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
        #done
    #done
#done 


# Store parameters
SUBTASK="unsupervisedmultidomainclassifier" #"featdist"
DATASET_NAME="pe_10percent_background_unsupervised"
LR=0.01
OUTPUT=".txt"


MODEL_NAME_=$"UNsupMultiDAN_"$DATASET_NAME"_"
#for gain in {"1.5","0.5","1.0","0.75","0.25","1.25","1.75","0.25","2"}  
#do
MODEL_NAME=$MODEL_NAME_$"dcgain"$"0.5_0.5_1.0"
MODEL_PATH=$"runs/detect/"$MODEL_NAME
OUTPUT_FILE=$MODEL_PATH$OUTPUT
python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --gains $"0.5",$"0.5",$"1.0"
python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done
