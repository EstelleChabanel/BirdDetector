#!/bin/bash




# on several Datasets
SUBTASK="detect"
LR=0.01
OUTPUT=".txt"
#{"pe_palmyra_10percentbkgd","pepf_10percent_background","pepf_palmyra_10percentbkgd"}
#for dataset in {"pepf_te_10percent_background"} # "te_mckellar_10percent_background","palm_mckellar_penguin_10percent_background"}  #"all_10percent_background_pfenobackgd","all_datasets_10percent_background","","",""}


# IOU grid search
dataset="pe_palm_mckellar_10percentbkgd"
iou=0.3
LR=0.01
#for dataset in {"palmyra_10percent_background","pe_10percent_background","pepf_10percent_background"}
#do
MODEL_NAME=$"YOLO_"$dataset
MODEL_PATH=$"runs/detect/"$MODEL_NAME
OUTPUT_FILE=$MODEL_PATH$OUTPUT
echo $MODEL_NAME
#python src/model/originaltrainer_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") >> $OUTPUT_FILE
#python src/model/originalevaluator_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --iou $(bc -l <<<"${iou}")
python src/model/additional_originalevaluator.py --model-name "$MODEL_NAME" 
#done



#python src/model/gridsearch_selection_.py --model-name "$MODEL_NAME_" --dataset-name "$dataset" --param "lr" --params $"0.0001","0.005","0.0085","0.01","0.025","0.05","0.075","0.1"}

# Learning Rate grid search
for dataset in {"pe_palmyra_10percentbkgd","mckellar_10percent_background"}
do
    for lr in {0.005,0.0085,0.01,0.025,0.05,0.075,0.1} #0.0001,0.005,0.0085,0.01,
    do
        MODEL_NAME=$"YOLO_"$dataset$"_lr"$lr
        MODEL_PATH=$"runs/detect/"$MODEL_NAME
        python src/model/originaltrainer_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --lr $(bc -l <<<"${lr}") 
    done
    MODEL_NAME_=$"YOLO_"$dataset$"_lr"
    python src/model/gridsearch_selection_.py --model-name "$MODEL_NAME_" --dataset-name "$dataset" --param $"lr" --params $"0.0001","0.005","0.0085","0.01","0.025","0.05","0.075","0.1"
done






dataset="all_datasets_minusHayesTerns_10percentbkgd"
#for iou in {0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
#do
MODEL_NAME=$"YOLO_"$dataset
MODEL_PATH=$"runs/detect/"$MODEL_NAME
OUTPUT_FILE=$MODEL_PATH$OUTPUT
echo $MODEL_NAME
#python src/model/originaltrainer_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") >> $OUTPUT_FILE
#python src/model/originalevaluator_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --iou $(bc -l <<<"${iou}")
#done



#dataset="alldatasets_minus_hayes"
#iou=0.1
#MODEL_NAME=$"YOLO_"$dataset
#MODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/originaltrainer_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") >> $OUTPUT_FILE
#python src/model/originalevaluator_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --iou $(bc -l <<<"${iou}")




# BIG MODEL
#DATASET_NAME=""
#MODEL_NAME=$"YOLO_alldatasets"
#MODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/originaltrainer_.py --model-name "$MODEL_NAME" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") >> $OUTPUT_FILE
#python src/model/originalevaluator_.py --model-name "$MODEL_NAME" --dataset-name "$DATASET_NAME" --iou-threshold "$IOU" >> $OUTPUT_FILE 
#done 




# Store parameters
#SUBTASK="detect"
#DATASET_NAME="pe_palmyra_10percentbkgd"
#LR=0.01
#OUTPUT=".txt"

# IOU grid search
##for iou in {0.5,0.6,0.7,0.8,0.9}  #0.1,0.2,0.3,0.4,
#do
 #   MODEL_NAME=$"YOLO_"$DATASET_NAME$"_iou_"$iou
 #   MODEL_PATH=$"runs/detect/"$MODEL_NAME
 #   OUTPUT_FILE=$MODEL_PATH$OUTPUT
 #   echo $MODEL_NAME
 #   python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${LR}") >> $OUTPUT_FILE
 #   python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --iou-threshold "$iou" >> $OUTPUT_FILE 
#done 

# Gridsearch LearningRate
#MODEL_NAME_="oldpfeifer_test_"
#for lr in {0.00001,0.00005,0.0005,0.0001,0.001,0.005,0.01}
#do
#    MODEL_NAME=$MODEL_NAME_$lr
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $OUTPUT_FILE
#    python src/model/original_trainer_.py --model-name "$MODEL_NAME" --dataset-name "$DATASET_NAME" --lr $(bc -l <<<"${lr}") --dcloss-gain $(bc -l <<<"${DC_LOSS_GAIN}") >> $OUTPUT_FILE
#    python src/model/original_evaluator_.py --model-name "$MODEL_NAME" --dataset-name "$DATASET_NAME"
#done 