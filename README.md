# BirdDetector
  
### *Master Thesis - Mathematics Section, Master in Computational Science and Engineering*
### Estelle Chabanel
    
  
This repository contains the code related to the research and experiments performed for my Master Thesis on *[Domain Adpatation for Birds Monitoring form aerial images](MasterThesis.pdf)*.   
  
More specifically, it implements distinct adversarial domain adaptation methods as an extension of the YOLOv8 framework for detection [1].  
  
  
## Installation  

This project is coded using *Pytohn 3.11.6*. Required packages can be installed from the [requirements.txt file](requirements.txt).
````
pip install -r requirements.txt
````

## Usage

On top of the existing task for detection, the YOLOv8 framework is extended with 8 new tasks corresponding to distinct domain adaptation architectures. Dinstinct architectures are described in details the report and are initiliazed with the follwoing conventions:

|  Domain adaptation architecture  |    Supervised subtask   |          Unuspervised subtask       |  
| -------------------------------- | ----------------------- | ----------------------------------- |
|        L2 norm minimization      |   `featuresdistance`    |     `unsupervisedfeaturesdistance`    |  
|     Single Domain Classifier     |   `domainclassifier`    |    `unsuperviseddomainclassifier`   |  
|     Multi-Domain Classifiers     | `multidomainclassifier` | `unsupervisedmultidomainclassifier` |  
| Multi-Features Domain Classifier |    `multifeaturesDC`    |     `unsupervisedmultifeatsDC`      |  

The user can run any of these domain adaptation architectures using the following command:
````
python src/model/trainer_.py --model-name "MODEL_NAME" --subtask "SUBTASK" --dataset-name "DATASET_NAME"
````

where the `model-name`, `dataset-name` need to be defined, the `subtask` parameters is chosen among one of the above.
The training script uses default training parameters defined in the [constants.py file](src/model/constants.py). The file also contains the source and target domain correspondance for the distinct datasets, and the path to the datasets and models also need to be modified in this file that need to be modified accoridng to usage.

The models are evaluated using the following command:
````
#python src/model/evaluator_.py --model-name "MODEL_NAME" --subtask "SUBTASK" --dataset-name "DATASET_NAME"
````
A default IoU threshold off 0.3 is used for model evaluation but can be modified using the argument --iou

The modified YOLOv8 Ultralytics [1] library is renamed to *yolo* and stored in the [yolo folder](yolo).

## References

<a id="1">[1]</a> 
[Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)), 2023
