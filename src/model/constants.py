import numpy as np


# Data
DATA_PATH = '/gpfs/gibbs/project/jetz/eec42/data/'
DATASETS_MAPPING = {'pe_palmyra_10percentbkgd': ['global_birds_penguins', 'global_birds_palmyra'],
                    'te_palm_10percent_background': ['global_birds_palmyra', 'terns_africa'],
                    'te_mckellar_10percent_background': ['global_birds_mckellar', 'terns_africa'],
                    'palm_mckellar_penguin_10percent_background': ['global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins'],
                    'all_datasets_10percent_background': ['global-bird-zenodo_pfeifer', 'global-bird-zenodo_palmyra', 'global-bird-zenodo_mckellar', 'global-bird-zenodo_penguins', 'global-bird-zenodo_poland', 'uav-waterfowl-thermal', 'hayes_albatross', 'terns_africa']
                    }

EVAL_DATASETS_MAPPING = {'pe_palmyra_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_palmyra']},
                         'te_palm_10percent_background': {'source': ['global_birds_palmyra', 'terns_africa']},
                         'te_mckellar_10percent_background': {'source': ['global_birds_mckellar', 'terns_africa']},
                         'palm_mckellar_penguin_10percent_background': {'source': ['global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins']},
                         'all_datasets_10percent_background': {'source': ['global-bird-zenodo_pfeifer', 'global-bird-zenodo_palmyra', 'global-bird-zenodo_mckellar', 'global-bird-zenodo_penguins', 'global-bird-zenodo_poland', 'uav-waterfowl-thermal', 'hayes_albatross', 'terns_africa']}
                         }


# Model
MODELS_PATH = 'runs/detect/'

# For training
NB_EPOCHS = 120 
BATCH_SIZE = 32
PATIENCE = 30
OPTIMIZER = 'Adam' # choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
TRAINING_IOU_THRESHOLD = 0.1

# For predictions
IOU_THRESHOLD = 0.1
CONF_THRESHOLD = 0.1

# For evaluation
NB_CONF_THRESHOLDS = 50
CONF_THRESHOLDS = np.linspace(0, 1, NB_CONF_THRESHOLDS) # CAREFUL: if you change that, don't forget to change calls to plot_confusion_matrix function
