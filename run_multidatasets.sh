#!/bin/bash



SUBTASK="unsuperviseddomainclassifier"
dataset="pe_10percent_background_unsupervised"
LR=0.01
DC_LOSS_GAIN=1.5
OUTPUT=".txt"


MODEL_NAME=$"UNsup_DAN_"$dataset
MODEL_PATH=$"runs/detect/"$MODEL_NAME
OUTPUT_FILE=$MODEL_PATH$OUTPUT
echo $MODEL_NAME
python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
 


# Store parameters
SUBTASK="domainclassifier"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=1.5
OUTPUT=".txt"


#dataset="pepf_palmyra_10percentbkgd"
#MODEL_NAME=$"DAN_"$dataset$"_test2"
#MODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
#done 

# Test on several Datasets     # pe_te_10percent_background
#for dataset in {"pepf_palmyra_10percentbkgd","pepf_te_10percent_background"} #{"pepf_palmyra_10percentbkgd","pe_te_10percent_background","pepf_te_10percent_background","te_palm_10percent_background"}
#do
 #   MODEL_NAME=$"DAN_"$dataset
 #   MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $MODEL_NAME
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
#done 



# Store parameters
SUBTASK="multidomainclassifier"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=0.5
OUTPUT=".txt"

# Test on several Datasets
#for dataset in {"pepf_palmyra_10percentbkgd","pepf_te_10percent_background","te_palm_10percent_background"}
#do
#    MODEL_NAME=$"multiDAN_"$dataset
 #   MODEL_PATH=$"runs/detect/"$MODEL_NAME
 #   OUTPUT_FILE=$MODEL_PATH$OUTPUT
 #   echo $MODEL_NAME
 #   python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
 #   python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
#done 


# Store parameters
SUBTASK="featuresdistance"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=5.0
OUTPUT=".txt"

# Test on several Datasets
#for dataset in {"pe_palmyra_10percentbkgd","pe_te_10percent_background"}
#do
#    MODEL_NAME=$"featdist_"$dataset
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $MODEL_NAME
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
#done 


# Store parameters
SUBTASK="detect"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=1.0
OUTPUT=".txt"


# Test on several Datasets
#for dataset in {"pepf_palmyra_10percentbkgd","te_palm_10percent_background"}
#do
#    MODEL_NAME=$"YOLO_"$dataset
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $MODEL_NAME
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
#done 

#dataset="pepf_te_10percent_background"
#MODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"



# Store parameters
#SUBTASK="featuresdistance"
#DATASET_NAME=""
#LR=0.01
#DC_LOSS_GAIN=5.0
#OUTPUT=".txt"


#dataset="pe_palmyra_10percentbkgd"
#MODEL_NAME=$"featdist_"$dataset$"_test3"
#MODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"


# Store parameters
#SUBTASK="featuresdistance"
#DATASET_NAME=""
#LR=0.01
#DC_LOSS_GAIN=5.0
#OUTPUT=".txt"


#dataset="pepf_palmyra_10percentbkgd"
#MODEL_NAME=$"featdist_"$dataset$"_test2"
#MODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"


# Store parameters
SUBTASK="multifeaturesDC"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=0.5
OUTPUT=".txt"


#dataset="pe_palmyra_10percentbkgd"
#MODEL_NAME=$"multiDC2_"$dataset$"_test_SiLU_2"
#MODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"




SUBTASK="multifeaturesDC"
DATASET_NAME=""
LR=0.01
DC_LOSS_GAIN=1.5
OUTPUT=".txt"

# Test on several Datasets     # pe_te_10percent_background
#for dataset in {"pepf_palmyra_10percentbkgd","pe_te_10percent_background","pepf_te_10percent_background"} #{"pepf_palmyra_10percentbkgd","pe_te_10percent_background","pepf_te_10percent_background","te_palm_10percent_background"}
#do
#    MODEL_NAME=$"multiDC2_"$dataset
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $MODEL_NAME
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$dataset"
#done 
