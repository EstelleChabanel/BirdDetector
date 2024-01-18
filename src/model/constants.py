import numpy as np


# Data
DATA_PATH = '/gpfs/gibbs/project/jetz/eec42/data/'
DATASETS_MAPPING = {'pe_palmyra_10percentbkgd': ['global_birds_penguins', 'global_birds_palmyra'],
                    'te_palm_10percent_background': ['global_birds_palmyra', 'terns_africa'],
                    'pe_te_10percent_background': ['global_birds_penguins', 'terns_africa'],
                    'te_mckellar_10percent_background': ['global_birds_mckellar', 'terns_africa'],
                    'palm_mckellar_penguin_10percent_background': ['global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins'],
                    'palm_hayes_10percent_background': ['global_birds_palmyra', 'hayes_albatross'],
                    'all_datasets_10percent_background': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'],
                    'all_10percent_background_pfenobackgd': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa']
                    }

EVAL_DATASETS_MAPPING = {'pe_palmyra_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_palmyra']},
                         'te_palm_10percent_background': {'source': ['global_birds_palmyra', 'terns_africa']},
                         'pe_te_10percent_background': {'source': ['global_birds_penguins', 'terns_africa']},
                         'te_mckellar_10percent_background': {'source': ['global_birds_mckellar', 'terns_africa']},
                         'palm_mckellar_penguin_10percent_background': {'source': ['global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins']},
                         'all_datasets_10percent_background': {'source': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'],},
                         'all_10percent_background_pfenobackgd': {'source': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'],}
                         }


# Model
MODELS_PATH = 'runs/detect/'

# For training
NB_EPOCHS = 200 #120 
BATCH_SIZE = 32
PATIENCE = 30
OPTIMIZER = 'SGD' # choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
TRAINING_IOU_THRESHOLD = 0.1

# For predictions
IOU_THRESHOLD = 0.1
CONF_THRESHOLD = 0.1

# For evaluation
NB_CONF_THRESHOLDS = 50
CONF_THRESHOLDS = np.linspace(0, 1, NB_CONF_THRESHOLDS) # CAREFUL: if you change that, don't forget to change calls to plot_confusion_matrix function
