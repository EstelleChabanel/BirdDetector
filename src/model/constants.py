import numpy as np


# Data
DATA_PATH = '/gpfs/gibbs/project/jetz/eec42/data/'
DATASETS_MAPPING = {'pepf_10percent_background':  {'datasets': ['global_birds_penguins', 'global_birds_pfeifer'], 'source': ''},
                    'pe_palmyra_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},
                    'pepf_palmyra_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},
                    'te_palm_10percent_background': {'datasets': ['global_birds_palmyra', 'terns_africa'], 'source': 'global_birds_palmyra'},
                    'pe_te_10percentbkgd': {'datasets': ['global_birds_penguins', 'terns_africa'], 'source': 'terns_africa'},
                    'pepf_te_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'terns_africa'], 'source': 'terns_africa'},
                    'poland_palmyra_10percent_background': {'datasets': ['global_birds_poland', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},

                    'all_datasets_10percent_background': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'], 'source': ''},
                    'all_10percent_background_pfenobackgd': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'], 'source': ''}
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
