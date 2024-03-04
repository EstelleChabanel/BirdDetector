# BirdDetector
  
### *Master Thesis - Mathematics Section, Master in Computational Science and Engineering*
### Estelle Chabanel
    
  
This repository contains the code related to the research and experiments performed for my Master Thesis on *[Domain Adpatation for Birds Monitoring form aerial images](MasterThesis.pdf)*.   
  
More specifically, it implements distinct adversarial domain adaptation methods as an extension of the YOLOv8 framework for detection [1].  
  
  
## Installation  

This project is coded using *Python 3.11.6*. Required packages can be installed from the [requirements.txt file](requirements.txt).
````
pip install -r requirements.txt
````

## Usage

On top of the existing task for detection, the YOLOv8 framework is extended with 8 new tasks corresponding to distinct domain adaptation architectures. Dinstinct architectures are described in details in the report and are initiliazed with the follwoing conventions:

|  Domain adaptation architecture  |    Supervised subtask   |          Unuspervised subtask       |  
| -------------------------------- | ----------------------- | ----------------------------------- |
|        L2 norm minimization      |   `featuresdistance`    |     `unsupervisedfeaturesdistance`    |  
|     Single Domain Classifier     |   `domainclassifier`    |    `unsuperviseddomainclassifier`   |  
|     Multi-Domain Classifiers     | `multidomainclassifier` | `unsupervisedmultidomainclassifier` |  
| Multi-Features Domain Classifier |    `multifeaturesDC`    |     `unsupervisedmultifeatsDC`      |  

The user can run any of these domain adaptation architectures using the following command:
````
python src/model/trainer_.py --model-name "MODEL_NAME" --subtask "SUBTASK" --dataset-name "DATASET_NAME" --default-param True
````

* where the `model-name` is a *string* that denotes the folder in which results will be saved, 
* `dataset-name` is a *string* and define the name of the folder storing the data, 
* the `subtask` parameters is chosen among one of the above.
* When initialized with True for the `default-param`, parameters are innitiliazed to the pred-defined optimum parameters corresponding to the chosen dataset. When running a new, unkown dataset, the parameters are iinitilaized to the default values of the YOLOv8 model.  
  
All parameters defined in the [constants.py file](src/model/constants.py). The file also contains the source and target domain correspondance for the distinct datasets, and the path to the datasets and models. These last ones need to be adapted to your own datasets and folder organization. Note that the organization of the of the data need to follow the one required by Ultralytics. For unsupervised training, the train folder needs to be divided into two subfolders, *source* and *target*.

The models are evaluated using the following command:
````
python src/model/evaluator_.py --model-name "MODEL_NAME" --subtask "SUBTASK" --dataset-name "DATASET_NAME" --iou IOU
````
When `iou` is not precised, a default NMS IoU threshold of 0.3 is used.    
   
Here are two examples to trian and evaluate the *Single Domain Classifier* architectures in both supervised adn unsupervised setup on the *Penguins* to *Palmyra* datasets, with the right path defined in the [constants.py file](src/model/constants.py) (in our case, the datasets stored in the `/gpfs/gibbs/pi/jetz/projects/` partition of the Yale cluster). Note that the datasets are different for both setups since the train folder does not contain target examples in the unsupervised setup.
````
python src/model/trainer_.py --model-name "singleDC_pe_palmyra_10percentbkgd" --subtask "domainclassifier" --dataset-name "pe_palmyra_10percentbkgd" --default-param True  
python src/model/evaluator_.py --model-name "singleDC_pe_palmyra_10percentbkgd" --subtask "domainclassifier" --dataset-name "pe_palmyra_10percentbkgd" --iou 0.3  
python src/model/trainer_.py --model-name "UNSUPsingleDC_pe_palmyra_10percentbkgd" --subtask "unsuperviseddomainclassifier" --dataset-name "pe_10percent_background_unsupervised" --default-param True  
python src/model/evaluator_.py --model-name "UNSUPsingleDC_pe_palmyra_10percentbkgd" --subtask "unsuperviseddomainclassifier" --dataset-name "pe_10percent_background_unsupervised" --iou 0.3  
````
    

The modified YOLOv8 Ultralytics [1] library is renamed to *yolo* and stored in the [yolo folder](yolo).


## References

<a id="1">[1]</a> 
[Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/), 2023
