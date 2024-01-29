#!/bin/bash


# Store parameters
SUBTASK="domainclassifier"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=1.5
OUTPUT=".txt"


for dataset in {"pe_palmyra_10percentbkgd","pepf_palmyra_10percentbkgd","pe_mckellar_10percentbkgd"} 
do
    MODEL_NAME=$"DAN_"$dataset
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $MODEL_NAME
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
done 


# Store parameters
SUBTASK="multidomainclassifier"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=0.5
OUTPUT=".txt"

for dataset in {"pe_palmyra_10percentbkgd","pepf_palmyra_10percentbkgd","pe_mckellar_10percentbkgd"} 
do
    MODEL_NAME=$"multiDAN_"$dataset
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $MODEL_NAME
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
done 



# Store parameters
SUBTASK="featuresdistance"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=5.0
OUTPUT=".txt"

for dataset in {"pe_palmyra_10percentbkgd","pepf_palmyra_10percentbkgd","pe_mckellar_10percentbkgd"} 
do
    MODEL_NAME=$"featdist_"$dataset
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $MODEL_NAME
    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
done 


# Store parameters
SUBTASK="multifeaturesDC"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=0.5
OUTPUT=".txt"

#for dataset in {"pe_mckellar_10percentbkgd","te_poland_10percentbkgd"}
#do
    #MODEL_NAME=$"DAN_"$dataset$
    #MODEL_PATH=$"runs/detect/"$MODEL_NAME
    #OUTPUT_FILE=$MODEL_PATH$OUTPUT
    #echo $MODEL_NAME
    #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
    #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
#done 




# ================ UNSUPERVISED ================ #

#SUBTASK="unsuperviseddomainclassifier"
#dataset="pe_10percent_background_unsupervised"
#LR=0.01
#DC_LOSS_GAIN=1.5
#OUTPUT=".txt"


#MODEL_NAME=$"UNsup_DAN_"$dataset$"_noval"
#MODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
 

#SUBTASK="unsupervisedmultidomainclassifier"
#dataset="pe_10percent_background_unsupervised"
#LR=0.01
#DC_LOSS_GAIN=0.5
#OUTPUT=".txt"

#MODEL_NAME=$"UNsup_multiDAN_"$dataset$"_noval"
#MODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
 