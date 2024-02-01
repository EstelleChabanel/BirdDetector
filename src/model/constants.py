import numpy as np


# Data
DATA_PATH = '/gpfs/gibbs/project/jetz/eec42/data/'
DATASETS_MAPPING = {'pepf_10percent_background': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer'], 'source': ''},
                    'palmyra_10percent_background': {'datasets': ['global_birds_palmyra'], 'source': ''},
                    'pe_10percent_background': {'datasets': ['global_birds_penguins'], 'source': ''},
                    'pe_10percent_background_unsupervised': {'datasets': ['global_birds_penguins', 'global_birds_palmyra'], 'source': ''},
                    'pe_palmyra_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},
                    'pepf_palmyra_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},
                    'pepfpol_palmyra_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_poland', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},
                    'pepfpol_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_poland'], 'source': ''},
                    'pepol_palmyra_datasets_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_poland', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},

                    'pe_mckellar_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_mckellar'], 'source': 'global_birds_mckellar'},
                    'te_poland_10percentbkgd': {'datasets': ['terns_africa', 'global_birds_poland'], 'source': 'global_birds_poland'},
                    'terns_10percentbkgd': {'datasets': ['terns_africa'], 'source': ''},
                    'poland_10percentbkgd': {'datasets': ['global_birds_poland'], 'source': ''},
                    'mckellar_10percentbkgd': {'datasets': ['global_birds_mckellar'], 'source': ''},

                    'te_palm_10percent_background': {'datasets': ['global_birds_palmyra', 'terns_africa'], 'source': 'global_birds_palmyra'},
                    'pe_te_10percent_background': {'datasets': ['global_birds_penguins', 'terns_africa'], 'source': 'terns_africa'},
                    'pepf_te_10percent_background': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'terns_africa'], 'source': 'terns_africa'},
                    'poland_palmyra_10percent_background': {'datasets': ['global_birds_poland', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},

                    'palmyra_mckellar_10percentbkgd': {'datasets': ['global_birds_palmyra', 'global_birds_mckellar'], 'source': 'global_birds_mckellar'},
                    'pe_palm_mckellar_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_palmyra', 'global_birds_mckellar'], 'source': ['global_birds_palmyra', 'global_birds_mckellar']},
                    
                    'all_datasets_minusHayesTerns_10percentbkgd_onall': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl'], 'source': ['global_birds_mckellar', 'uav_thermal_waterfowl']},
                    'all_datasets_minusHayesTerns_10percentbkgd': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl'], 'source': ''},
                    'alldatasets_allbckgd': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'terns_africa', 'hayes_albatross'], 'source': ''},
                    'alldatasets_minus_hayes': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'terns_africa'], 'source': ''},
                    'all_datasets_10percent_background': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'], 'source': 'global_birds_mckellar'},
                    'all_10percent_background_pfenobackgd': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'], 'source': ''}
                    }

EVAL_DATASETS_MAPPING = {'pepf_10percent_background': {'source': ['global_birds_penguins', 'global_birds_pfeifer'], 'untrained_target': ['global_birds_palmyra']},
                         'palmyra_10percent_background': {'source': ['global_birds_palmyra'], 'untrained_target': ['global_birds_pfeifer', 'global_birds_penguins', 'global_birds_poland']},
                         'pe_10percent_background': {'source': ['global_birds_penguins'], 'untrained_target': ['global_birds_palmyra', 'global_birds_mckellar']},
                         'pe_10percent_background_unsupervised': {'source': ['global_birds_penguins', 'global_birds_palmyra'], 'untrained_target': []},
                         'pe_palmyra_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_palmyra'], 'untrained_target': ['global_birds_mckellar', 'global_birds_poland']},
                         'pepf_palmyra_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_palmyra']},
                         'pepfpol_palmyra_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_poland', 'global_birds_palmyra']},
                         'pepfpol_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_poland'], 'untrained_target': ['global_birds_palmyra']},
                         'pepol_palmyra_datasets_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_poland', 'global_birds_palmyra']},

                         'pe_mckellar_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_mckellar']},
                         'te_poland_10percentbkgd': {'source': ['terns_africa', 'global_birds_poland']},
                         'terns_10percentbkgd': {'source': ['terns_africa'], 'untrained_target': ['global_birds_poland']},
                         'poland_10percentbkgd': {'source': ['global_birds_poland'], 'untrained_target': ['terns_africa']},
                         'mckellar_10percentbkgd': {'source': ['global_birds_mckellar'], 'untrained_target': ['global_birds_penguins']},

                         'te_palm_10percent_background': {'source': ['global_birds_palmyra', 'terns_africa']},
                         'pe_te_10percent_background': {'source': ['global_birds_penguins', 'terns_africa']},
                         'pepf_te_10percent_background': {'source': ['global_birds_penguins', 'global_birds_pfeifer', 'terns_africa']},
                         'te_mckellar_10percent_background': {'source': ['global_birds_mckellar', 'terns_africa']},
                         'palm_mckellar_penguin_10percent_background': {'source': ['global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins']},

                         'palmyra_mckellar_10percentbkgd': {'source': ['global_birds_palmyra', 'global_birds_mckellar']},
                         'pe_palm_mckellar_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_palmyra', 'global_birds_mckellar']},

                    
                         'all_datasets_minusHayesTerns_10percentbkgd_onall': {'source': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl']},
                         'all_datasets_minusHayesTerns_10percentbkgd': {'source': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl']},
                         'alldatasets_allbckgd': {'source': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'terns_africa', 'hayes_albatross'],},
                         'alldatasets_minus_hayes': {'source': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'terns_africa']},
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
MATCH_IOU_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.3
CONF_THRESHOLD = 0.1

# For evaluation
NB_CONF_THRESHOLDS = 20
CONF_THRESHOLDS = np.linspace(0, 1, NB_CONF_THRESHOLDS) # CAREFUL: if you change that, don't forget to change calls to plot_confusion_matrix function
