#!/bin/bash

# SAVE PREDICTIONS
SUBTASK="domainclassifier"
DATASET_NAME="poland_mckellar_10percentbkgd"
MODEL_NAME=$"DAN_poland_mckellar_10percentbkgd_dcgain1.0"
#python src/model/predictor_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"

SUBTASK="multidomainclassifier"
DATASET_NAME="poland_mckellar_10percentbkgd"
MODEL_NAME=$"multiDAN_poland_mckellar_10percentbkgd_dcgain1.0_1.0_1.0"
#python src/model/predictor_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"

SUBTASK="multifeaturesDC"
DATASET_NAME="poland_mckellar_10percentbkgd"
MODEL_NAME=$"multifeatsDC_poland_mckellar_10percentbkgd_dcgain1.5"
#python src/model/predictor_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"

SUBTASK="featuresdistance"
DATASET_NAME="poland_mckellar_10percentbkgd"
MODEL_NAME=$"FeatsDist_poland_mckellar_10percentbkgd_dcgain0.25"
python src/model/predictor_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"



# SUPERVISED DA & GIRD SEARCHES
## Gridsearch DC loss gain
SUBTASK="domainclassifier"
DATASET_NAME="pe_palmyra_10percentbkgd"
#MODEL_NAME=$"DAN_pe_palmyra_10percentbkgd_dcgain1.5"
MODEL_NAME_=$"DAN_"$DATASET_NAME$"_dcgain"
for gain in {1.5,1.0,0.5}  #0.1,0.5,   0.5,0.75,2,3
do
    MODEL_NAME=$MODEL_NAME_$gain$"_morepatience"
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --dcloss-gain $(bc -l <<<"${gain}") --default-param True #>> $OUTPUT_FILE
    #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 


SUBTASK="instanceDC"
DATASET_NAME="pe_palmyra_10percentbkgd"
#MODEL_NAME=$"DAN_pe_palmyra_10percentbkgd_dcgain1.5"
MODEL_NAME_=$"instanceDAN_"$DATASET_NAME$"_dcgain"
for gain in {1.5,1.0,0.5}  #0.1,0.5,   0.5,0.75,2,3
do
    MODEL_NAME=$MODEL_NAME_$gain$"_morepatience"
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --dcloss-gain $(bc -l <<<"${gain}") --default-param True #>> $OUTPUT_FILE
    #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 


# Store parameters
SUBTASK="multifeaturesDC"
DATASET_NAME="poland_mckellar_10percentbkgd"

# Gridsearch DC loss gain
MODEL_NAME_=$"multifeatsDC_"$DATASET_NAME$"_dcgain"
for gain in {1.5,1.0,0.5} #{0.25,0.5,0.75,1.0,1.5,2,3,5,10}
do
    MODEL_NAME=$MODEL_NAME_$gain
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
    #OUTPUT_FILE=$MODEL_PATH$OUTPUT
    #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --default-param True --dcloss-gain $(bc -l <<<"${gain}") 
    #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 


# Store parameters
SUBTASK="featuresdistance"
DATASET_NAME="poland_mckellar_10percentbkgd"

# Gridsearch DC loss gain
MODEL_NAME_=$"FeatsDist_"$DATASET_NAME$"_dcgain"
#for gain in {0.25,0.5,1.0,1.5} #{0.25,0.5,0.75,1.0,1.5,2,3,5,10}
#do
#    MODEL_NAME=$MODEL_NAME_$gain
#    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    #OUTPUT_FILE=$MODEL_PATH$OUTPUT
#    python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --default-param True --dcloss-gain $(bc -l <<<"${gain}") 
#    python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
#done 


# Store parameters
SUBTASK="multidomainclassifier" #"featdist"
DATASET_NAME="poland_mckellar_10percentbkgd"
#LR=0.01
#OUTPUT=".txt"

# Gridsearch DC loss gain
MODEL_NAME_=$"multiDAN_"$DATASET_NAME"_"
for gain1 in {"1.0","1.5"} #,"0.75","1.25","1.75","0.25","2"}  
do
    for gain2 in {"1.0","1.5"}   # "1.25","0.25","1.75","2","0.75",
    do
        for gain3 in {"1.0","1.5"}   # ,"1.25","1.75","2","0.75","0.25"  #{"0.25","0.5","0.75","1.0","1.25","1.5","1.75","2"}
        do
            MODEL_NAME=$MODEL_NAME_$"dcgain"$gain1$"_"$gain2$"_"$gain3
            MODEL_PATH=$"runs/detect/"$MODEL_NAME
 #           OUTPUT_FILE=$MODEL_PATH$OUTPUT
            #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --gains $gain1,$gain2,$gain3 --default-param True #>> $OUTPUT_FILE
            #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
        done
    done
done 





# Store parameters
SUBTASK="unsuperviseddomainclassifier"
DATASET_NAME="poland_mckellar_unsupervised"

# Gridsearch DC loss gain
MODEL_NAME_=$"UNsupDAN_"$DATASET_NAME"_"
for gain in {1.5,0.5,1.0} #{0.25,0.5,0.75,1.0,1.5,2,3,5,10}
do
    MODEL_NAME=$MODEL_NAME_$"dcgain"$gain$"_morepatience"
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
#    OUTPUT_FILE=$MODEL_PATH$OUTPUT
    #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --dcloss-gain $(bc -l <<<"${gain}") --default-param True 
    #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 



# Store parameters
SUBTASK="unsupervisedmultidomainclassifier" #"featdist"
DATASET_NAME="poland_mckellar_unsupervised"   #"pe_10percent_background_unsupervised_moretarget"
LR=0.01
OUTPUT=".txt"


MODEL_NAME_=$"UNsupMultiDAN_"$DATASET_NAME"_"

MODEL_NAME=$MODEL_NAME_$"dcgain1.0_1.0_1.0"
MODEL_PATH=$"runs/detect/"$MODEL_NAME
#python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --gains $gain1,$gain2,$gain3 --default-param True
#python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
        
for gain1 in {"0.5","1.5"}  
do
    for gain2 in {"1.0","1.5"}
    do 
        for gain3 in {"1.0","1.5"}
        do
            MODEL_NAME=$MODEL_NAME_$"dcgain"$gain1$"_"$gain2$"_"$gain3
            MODEL_PATH=$"runs/detect/"$MODEL_NAME
            OUTPUT_FILE=$MODEL_PATH$OUTPUT
            #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --gains $gain1,$gain2,$gain3 --default-param True
            #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
        done
    done
done
#





# Store parameters
SUBTASK="unsupervisedfeaturesdistance"
DATASET_NAME="poland_mckellar_unsupervised"
#LR=0.01
OUTPUT=".txt"

# Gridsearch DC loss gain
MODEL_NAME_=$"UNsupFeatDist_"$DATASET_NAME"_"
for gain in {1.0,1.5} #{0.25,0.5,0.75,1.0,1.5,2,3,5,10}
do
    MODEL_NAME=$MODEL_NAME_$"dcgain"$gain
    MODEL_PATH=$"runs/detect/"$MODEL_NAME
 #   OUTPUT_FILE=$MODEL_PATH$OUTPUT
    #python src/model/trainer_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME" --dcloss-gain $(bc -l <<<"${gain}") --default-param True
    #python src/model/evaluator_.py --model-name "$MODEL_NAME" --subtask "$SUBTASK" --dataset-name "$DATASET_NAME"
done 
