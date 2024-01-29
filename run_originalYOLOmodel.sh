#!/bin/bash

SUBTASK="detect"
iou=0.3
LR=0.01
OUTPUT=".txt"

#for dataset in {"te_poland_10percentbkgd","terns_10percentbkgd","poland_10percentbkgd"} #,"mckellar_10percentbkgd"} #"pe_mckellar_10percentbkgd","te_poland_10percentbkgd"}
#do
 #   MODEL_NAME=$"YOLO_"$dataset
 #   MODEL_PATH=$"runs/detect/"$MODEL_NAME
 #   OUTPUT_FILE=$MODEL_PATH$OUTPUT
 #   echo $MODEL_NAME
    #python src/model/originaltrainer_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") >> $OUTPUT_FILE
 #   python src/model/originalevaluator_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --iou $(bc -l <<<"${iou}")
#done

#dataset="pepf_palmyra_10percentbkgd"
#MODEL_NAME=$"YOLO_"$dataset
#M#ODEL_PATH=$"runs/detect/"$MODEL_NAME
#OUTPUT_FILE=$MODEL_PATH$OUTPUT
#echo $MODEL_NAME
#python src/model/originaltrainer_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") >> $OUTPUT_FILE
#python src/model/originalevaluator_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --iou $(bc -l <<<"${iou}")


#for dataset in {"terns_10percentbkgd","poland_10percentbkgd"}
#do
#    MODEL_NAME=$"YOLO_"$dataset
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    echo $MODEL_NAME
#    python src/model/originaltrainer_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") >> $OUTPUT_FILE
#    python src/model/originalevaluator_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --iou $(bc -l <<<"${iou}")
#done


# IOU grid search
dataset="all_datasets_minusHayesTerns_10percentbkgd_onall"
iou=0.3
for iou in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
do
    MODEL_NAME=$"YOLO_"$dataset$"_oldconfig_semi_test2"
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    echo $MODEL_NAME
    python src/model/originaltrainer_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --lr $(bc -l <<<"${LR}") >> $OUTPUT_FILE
    python src/model/originalevaluator_.py --model-name "$MODEL_NAME" --dataset-name "$dataset" --iou $(bc -l <<<"${iou}")
done

#for iou in {0.6,0.8,0.9,1.0}
#do
 #   MODEL_NAME=$"YOLO_"$dataset
  #  MODEL_PATH=$"runs/detect/"$MODEL_NAME
   # OUTPUT_FILE=$MODEL_PATH$OUTPUT
    #echo $MODEL_NAME
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