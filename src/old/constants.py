import numpy as np


# Data
DATA_PATH = '/gpfs/gibbs/project/jetz/eec42/data/'
DATASETS_MAPPING_ = {'pepf_10percent_background': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer'], 'source': ''},
                    'palmyra_10percent_background': {'datasets': ['global_birds_palmyra'], 'source': ''},
                    'pe_10percent_background': {'datasets': ['global_birds_penguins'], 'source': ''},
                    'pe_10percent_background_unsupervised': {'datasets': ['global_birds_penguins', 'global_birds_palmyra'], 'source': ''},
                    'pe_10percent_background_unsupervised_moretarget': {'datasets': ['global_birds_penguins', 'global_birds_palmyra'], 'source': ''},
                    'pe_palmyra_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},
                    'pe_palmyra_unsup': {'datasets': ['global_birds_penguins', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},

                    'pepf_palmyra_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},
                    'pepfpol_palmyra_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_poland', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},
                    'pepfpol_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_poland'], 'source': ''},
                    'pepol_palmyra_datasets_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_poland', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},

                    'pe_mckellar_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_mckellar'], 'source': 'global_birds_mckellar'},
                    'te_poland_10percentbkgd': {'datasets': ['terns_africa', 'global_birds_poland'], 'source': 'global_birds_poland'},
                    'terns_10percentbkgd': {'datasets': ['terns_africa'], 'source': ''},
                    'poland_10percentbkgd': {'datasets': ['global_birds_poland'], 'source': ''},
                    'mckellar_10percentbkgd': {'datasets': ['global_birds_mckellar'], 'source': ''},
                    'pe_mckellar_10percentbkgd_unsupervised': {'datasets': ['global_birds_penguins', 'global_birds_mckellar'], 'source': 'global_birds_mckellar'},

                    'te_palm_10percent_background': {'datasets': ['global_birds_palmyra', 'terns_africa'], 'source': 'global_birds_palmyra'},
                    'pe_te_10percent_background': {'datasets': ['global_birds_penguins', 'terns_africa'], 'source': 'terns_africa'},
                    'pepf_te_10percent_background': {'datasets': ['global_birds_penguins', 'global_birds_pfeifer', 'terns_africa'], 'source': 'terns_africa'},
                    'poland_palmyra_10percent_background': {'datasets': ['global_birds_poland', 'global_birds_palmyra'], 'source': 'global_birds_palmyra'},

                    'palmyra_mckellar_10percentbkgd': {'datasets': ['global_birds_palmyra', 'global_birds_mckellar'], 'source': 'global_birds_mckellar'},
                    'pe_palm_mckellar_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_palmyra', 'global_birds_mckellar'], 'source': ['global_birds_palmyra', 'global_birds_mckellar']},
                    'palmyra_mckellar_unsupervised': {'datasets': ['global_birds_palmyra', 'global_birds_mckellar'], 'source': 'global_birds_palmyra'},
                    
                    'poland_mckellar_10percentbkgd': {'datasets': ['global_birds_poland', 'global_birds_mckellar'], 'source': 'global_birds_mckellar'},
                    'poland_mckellar_unsupervised': {'datasets': ['global_birds_poland', 'global_birds_mckellar'], 'source': 'global_birds_mckellar'},

                    'terns_mckellar_unsupervised': {'datasets': ['terns_africa', 'global_birds_mckellar'], 'source': 'global_birds_mckellar'},

                    'all_datasets_minusHayesTerns_10percentbkgd_onall': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl'], 'source': ['global_birds_mckellar', 'uav_thermal_waterfowl']},
                    'all_datasets_minusHayesTerns_10percentbkgd': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl'], 'source': ''},
                    'alldatasets_allbckgd': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'terns_africa', 'hayes_albatross'], 'source': ''},
                    'alldatasets_minus_hayes': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'terns_africa'], 'source': ''},
                    'all_datasets_10percent_background': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'], 'source': 'global_birds_mckellar'},
                    'all_10percent_background_pfenobackgd': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'], 'source': ''}
                    }

EVAL_DATASETS_MAPPING = {'pepf_10percent_background': {'source': ['global_birds_penguins', 'global_birds_pfeifer'], 'untrained_target': ['global_birds_palmyra']},
                         'palmyra_10percent_background': {'source': ['global_birds_palmyra'], 'untrained_target': ['global_birds_pfeifer', 'global_birds_penguins', 'global_birds_poland']},
                         'pe_10percent_background': {'source': ['global_birds_penguins'], 'untrained_target': ['global_birds_palmyra', 'global_birds_palmyra_train', 'global_birds_mckellar']},
                         'pe_10percent_background_unsupervised': {'source': ['global_birds_penguins', 'global_birds_palmyra'], 'untrained_target': []},
                         'pe_10percent_background_unsupervised_moretarget': {'source': ['global_birds_palmyra', 'global_birds_palmyra_train', 'global_birds_palmyra_val', 'global_birds_penguins'], 'untrained_target': []},

                         'pe_palmyra_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_palmyra']}, #, 'untrained_target': ['global_birds_mckellar', 'global_birds_poland']},
                         'pe_palmyra_unsup': {'source': ['global_birds_penguins', 'global_birds_palmyra']},

                         'pepf_palmyra_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_palmyra']},
                         'pepfpol_palmyra_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_poland', 'global_birds_palmyra']},
                         'pepfpol_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_pfeifer', 'global_birds_poland'], 'untrained_target': ['global_birds_palmyra']},
                         'pepol_palmyra_datasets_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_poland', 'global_birds_palmyra']},

                         'pe_mckellar_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_mckellar']},
                         'te_poland_10percentbkgd': {'source': ['terns_africa', 'global_birds_poland']},
                         'terns_10percentbkgd': {'source': ['terns_africa'], 'untrained_target': ['global_birds_poland']},
                         'poland_10percentbkgd': {'source': ['global_birds_poland'], 'untrained_target': ['global_birds_mckellar']},
                         'mckellar_10percentbkgd': {'source': ['global_birds_mckellar'], 'untrained_target': ['global_birds_poland', 'global_birds_penguins']},
                         'pe_mckellar_10percentbkgd_unsupervised': {'source': ['global_birds_penguins', 'global_birds_mckellar', 'global_birds_mckellar_val', 'global_birds_mckellar_train']},

                         'te_palm_10percent_background': {'source': ['global_birds_palmyra', 'terns_africa']},
                         'pe_te_10percent_background': {'source': ['global_birds_penguins', 'terns_africa']},
                         'pepf_te_10percent_background': {'source': ['global_birds_penguins', 'global_birds_pfeifer', 'terns_africa']},
                         'te_mckellar_10percent_background': {'source': ['global_birds_mckellar', 'terns_africa']},
                         'palm_mckellar_penguin_10percent_background': {'source': ['global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins']},

                         'palmyra_mckellar_10percentbkgd': {'source': ['global_birds_palmyra', 'global_birds_mckellar']},
                         'pe_palm_mckellar_10percentbkgd': {'source': ['global_birds_penguins', 'global_birds_palmyra', 'global_birds_mckellar']},
                         'palmyra_mckellar_unsupervised': {'source': ['global_birds_palmyra', 'global_birds_mckellar', 'global_birds_mckellar_val', 'global_birds_mckellar_train']},
                    
                         'terns_mckellar_unsupervised': {'source': ['terns_africa', 'global_birds_mckellar', 'global_birds_mckellar_val', 'global_birds_mckellar_train']},

                         'poland_mckellar_10percentbkgd': {'source': ['global_birds_poland', 'global_birds_mckellar']},
                         'poland_mckellar_unsupervised': {'source': ['global_birds_poland', 'global_birds_mckellar'], 'untrained_target': ['global_birds_mckellar_train']},

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
NB_EPOCHS = 3 #200 #200 #120 
BATCH_SIZE = 32
PATIENCE = 30
OPTIMIZER = 'SGD' # choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
TRAINING_IOU_THRESHOLD = 0.1
DEFAULT_LOSS_GAIN = {'domainclassifier': 1.5, 'unsuperviseddomainclassifier': 1.5,
                     'multidomainclassifier': [0.5,0.5,1.0], 'unsupervisedmultidomainclassifier': [0.5,0.5,1.0],
                     'featuresdistance': 0.25, 'unsupervisedfeaturesdistance': 0.25,
                     'multifeaturesDC': 0.5, 'unsupervisedmultifeatsDC': 1.5
                    } # TODO: UPDATE

DEFAULT_PARAM_SET_ = {'default': {'lr': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 
                                 'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
                                 'degrees': 90, 'scale': 0.5, 'fliplr': 0.5, 'flipud': 0.5, 'scale': 0.5,
                                 'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
                                 },
                     'pe_palmyra_10percentbkgd': {'lr': 0.00979, 'lrf': 0.01, 'momentum': 0.94971, 'weight_decay': 0.00048, 
                                 'box': 8.23357, 'cls': 0.53081, 'dfl': 1.72224,
                                 'degrees': 44.72491, 'scale': 0.47999, 'fliplr': 0.47555, 'flipud': 0.48174,
                                 'hsv_h': 0.0146, 'hsv_s': 0.7305, 'hsv_v': 0.39509,
                                 },
                     'OLD_pe_10percent_background': {'lr': 0.00979, 'lrf': 0.01, 'momentum': 0.94971, 'weight_decay': 0.00048, 
                                 'box': 8.23357, 'cls': 0.53081, 'dfl': 1.72224,
                                 'degrees': 44.72491, 'scale': 0.47999, 'fliplr': 0.47555, 'flipud': 0.48174,
                                 'hsv_h': 0.0146, 'hsv_s': 0.7305, 'hsv_v': 0.39509,
                                 },
                     'pe_10percent_background_unsupervised': {'lr': 0.00979, 'lrf': 0.01, 'momentum': 0.94971, 'weight_decay': 0.00048, 
                                 'box': 8.23357, 'cls': 0.53081, 'dfl': 1.72224,
                                 'degrees': 44.72491, 'scale': 0.47999, 'fliplr': 0.47555, 'flipud': 0.48174,
                                 'hsv_h': 0.0146, 'hsv_s': 0.7305, 'hsv_v': 0.39509,
                                 },
                     'palmyra_10percent_background': {'lr': 0.00979, 'lrf': 0.01, 'momentum': 0.94971, 'weight_decay': 0.00048, 
                                 'box': 8.23357, 'cls': 0.53081, 'dfl': 1.72224,
                                 'degrees': 44.72491, 'scale': 0.47999, 'fliplr': 0.47555, 'flipud': 0.48174,
                                 'hsv_h': 0.0146, 'hsv_s': 0.7305, 'hsv_v': 0.39509,
                                 },
                     'palmyra_mckellar_unsupervised': {'lr': 0.00979, 'lrf': 0.01, 'momentum': 0.94971, 'weight_decay': 0.00048, 
                                 'box': 8.23357, 'cls': 0.53081, 'dfl': 1.72224,
                                 'degrees': 44.72491, 'scale': 0.47999, 'fliplr': 0.47555, 'flipud': 0.48174,
                                 'hsv_h': 0.0146, 'hsv_s': 0.7305, 'hsv_v': 0.39509,
                                 },
                     'poland_mckellar_10percentbkgd': {'lr': 0.00691, 'lrf': 0.01276, 'momentum': 0.937, 'weight_decay': 0.00058, 
                                 'box': 7.5, 'cls': 0.47651, 'dfl': 1.463,
                                 'degrees': 44.82499, 'scale': 0.4757, 'fliplr': 0.58144, 'flipud': 0.58409,
                                 'hsv_h': 0.01499, 'hsv_s': 0.7, 'hsv_v': 0.30742,
                                 },
                     'poland_10percentbkgd': {'lr': 0.00691, 'lrf': 0.01276, 'momentum': 0.937, 'weight_decay': 0.00058, 
                                 'box': 7.5, 'cls': 0.47651, 'dfl': 1.463,
                                 'degrees': 44.82499, 'scale': 0.4757, 'fliplr': 0.58144, 'flipud': 0.58409,
                                 'hsv_h': 0.01499, 'hsv_s': 0.7, 'hsv_v': 0.30742,
                                 },
                     'OLD_mckellar_10percentbkgd': {'lr': 0.00691, 'lrf': 0.01276, 'momentum': 0.937, 'weight_decay': 0.00058, 
                                 'box': 7.5, 'cls': 0.47651, 'dfl': 1.463,
                                 'degrees': 44.82499, 'scale': 0.4757, 'fliplr': 0.58144, 'flipud': 0.58409,
                                 'hsv_h': 0.01499, 'hsv_s': 0.7, 'hsv_v': 0.30742,
                                 },
                     'poland_mckellar_unsupervised': {'lr': 0.00691, 'lrf': 0.01276, 'momentum': 0.937, 'weight_decay': 0.00058, 
                                 'box': 7.5, 'cls': 0.47651, 'dfl': 1.463,
                                 'degrees': 44.82499, 'scale': 0.4757, 'fliplr': 0.58144, 'flipud': 0.58409,
                                 'hsv_h': 0.01499, 'hsv_s': 0.7, 'hsv_v': 0.30742,
                                 },
                     'pe_mckellar_10percentbkgd': {'lr': 0.01107, 'lrf': 0.01096, 'momentum': 0.98, 'weight_decay': 0.00049, 
                                 'box': 6.84666, 'cls': 0.57734, 'dfl': 1.35745,
                                 'degrees': 34.17275, 'scale': 0.39214, 'fliplr': 0.44461, 'flipud': 0.38164,
                                 'hsv_h': 0.0141, 'hsv_s': 0.70406, 'hsv_v': 0.35903,
                                 },
                     'pe_10percent_background': {'lr': 0.01107, 'lrf': 0.01096, 'momentum': 0.98, 'weight_decay': 0.00049, 
                                 'box': 6.84666, 'cls': 0.57734, 'dfl': 1.35745,
                                 'degrees': 34.17275, 'scale': 0.39214, 'fliplr': 0.44461, 'flipud': 0.38164,
                                 'hsv_h': 0.0141, 'hsv_s': 0.70406, 'hsv_v': 0.35903,
                                 },
                     'mckellar_10percentbkgd': {'lr': 0.01107, 'lrf': 0.01096, 'momentum': 0.98, 'weight_decay': 0.00049, 
                                 'box': 6.84666, 'cls': 0.57734, 'dfl': 1.35745,
                                 'degrees': 34.17275, 'scale': 0.39214, 'fliplr': 0.44461, 'flipud': 0.38164,
                                 'hsv_h': 0.0141, 'hsv_s': 0.70406, 'hsv_v': 0.35903,
                                 },
                     'pe_mckellar_10percentbkgd_unsupervised' : {'lr': 0.01107, 'lrf': 0.01096, 'momentum': 0.98, 'weight_decay': 0.00049, 
                                 'box': 6.84666, 'cls': 0.57734, 'dfl': 1.35745,
                                 'degrees': 34.17275, 'scale': 0.39214, 'fliplr': 0.44461, 'flipud': 0.38164,
                                 'hsv_h': 0.0141, 'hsv_s': 0.70406, 'hsv_v': 0.35903,
                                 },                  
                    }

PRETRAINED_MODELS = {'poland_mckellar_unsupervised': 'YOLO_poland_10percentbkgd',
                         'pe_palmyra_10percentbkgd': 'YOLO_pe_10percent_background'}

DEFAULT_PARAM_SET = {'lr': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 
                    'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
                    'degrees': 90, 'scale': 0.5, 'fliplr': 0.5, 'flipud': 0.5, 'scale': 0.5,
                    'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
                    },

DATASETS_MAPPING = {'all_datasets_10percent_background': {'datasets': ['global_birds_pfeifer', 'global_birds_palmyra', 'global_birds_mckellar', 'global_birds_penguins', 'global_birds_poland', 'uav_thermal_waterfowl', 'hayes_albatross', 'terns_africa'],
                                                           'source': '',
                                                           'default_loss_gains': {},
                                                           'param':  {'lr': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 
                                                                        'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
                                                                        'degrees': 90, 'scale': 0.5, 'fliplr': 0.5, 'flipud': 0.5, 'scale': 0.5,
                                                                        'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
                                                                        },
                                                           'pretrained_model': '',
                                                           },
                    'pe_palmyra_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_palmyra'], 
                                                  'source': 'global_birds_palmyra',
                                                  'default_loss_gains': {'domainclassifier': 1.5, 'unsuperviseddomainclassifier': 1.5,
                                                                            'multidomainclassifier': [0.5,0.5,1.0], 'unsupervisedmultidomainclassifier': [0.5,0.5,1.0],
                                                                            'featuresdistance': 0.25, 'unsupervisedfeaturesdistance': 0.25,
                                                                            'multifeaturesDC': 0.5, 'unsupervisedmultifeatsDC': 0.5
                                                                        },
                                                  'param': {'lr': 0.00979, 'lrf': 0.01, 'momentum': 0.94971, 'weight_decay': 0.00048, 
                                                            'box': 8.23357, 'cls': 0.53081, 'dfl': 1.72224,
                                                            'degrees': 44.72491, 'scale': 0.47999, 'fliplr': 0.47555, 'flipud': 0.48174,
                                                            'hsv_h': 0.0146, 'hsv_s': 0.7305, 'hsv_v': 0.39509,
                                                            },
                                                'pretrained_model': '',
                                                },
                    'pe_10percent_background_unsupervised': {'datasets': ['global_birds_penguins', 'global_birds_palmyra'], 
                                                  'source': 'global_birds_palmyra',
                                                  'default_loss_gains': {'domainclassifier': 1.5, 'unsuperviseddomainclassifier': 1.5,
                                                                            'multidomainclassifier': [0.5,0.5,1.0], 'unsupervisedmultidomainclassifier': [0.5,0.5,1.0],
                                                                            'featuresdistance': 0.25, 'unsupervisedfeaturesdistance': 0.25,
                                                                            'multifeaturesDC': 0.5, 'unsupervisedmultifeatsDC': 0.5
                                                                        },
                                                  'param': {'lr': 0.00979, 'lrf': 0.01, 'momentum': 0.94971, 'weight_decay': 0.00048, 
                                                            'box': 8.23357, 'cls': 0.53081, 'dfl': 1.72224,
                                                            'degrees': 44.72491, 'scale': 0.47999, 'fliplr': 0.47555, 'flipud': 0.48174,
                                                            'hsv_h': 0.0146, 'hsv_s': 0.7305, 'hsv_v': 0.39509,
                                                            },
                                                    'pretrained_model': 'YOLO_pe_10percent_background',
                                                },
                     'poland_mckellar_10percentbkgd': {'datasets': ['global_birds_poland', 'global_birds_mckellar'],
                                                       'source': 'global_birds_mckellar',
                                                       'default_loss_gains': {'domainclassifier': 1.0, 'unsuperviseddomainclassifier': 1.0,
                                                                            'multidomainclassifier': [1.0,1.0,1.0], 'unsupervisedmultidomainclassifier': [1.0,1.0,1.0],
                                                                            'featuresdistance': 0.25, 'unsupervisedfeaturesdistance': 0.25,
                                                                            'multifeaturesDC': 1.5, 'unsupervisedmultifeatsDC': 1.5
                                                                            },
                                                        'param': {'lr': 0.00691, 'lrf': 0.01276, 'momentum': 0.937, 'weight_decay': 0.00058, 
                                                                    'box': 7.5, 'cls': 0.47651, 'dfl': 1.463,
                                                                    'degrees': 44.82499, 'scale': 0.4757, 'fliplr': 0.58144, 'flipud': 0.58409,
                                                                    'hsv_h': 0.01499, 'hsv_s': 0.7, 'hsv_v': 0.30742,
                                                                    },
                                                        'pretrained_model': '',
                                                        },
                     'poland_mckellar_unsupervised': {'datasets': ['global_birds_poland', 'global_birds_mckellar'],
                                                       'source': 'global_birds_mckellar',
                                                       'default_loss_gains': {'domainclassifier': 1.0, 'unsuperviseddomainclassifier': 1.0,
                                                                            'multidomainclassifier': [1.0,1.0,1.0], 'unsupervisedmultidomainclassifier': [1.0,1.0,1.0],
                                                                            'featuresdistance': 0.25, 'unsupervisedfeaturesdistance': 0.25,
                                                                            'multifeaturesDC': 1.5, 'unsupervisedmultifeatsDC': 1.5
                                                                            },
                                                        'param': {'lr': 0.00691, 'lrf': 0.01276, 'momentum': 0.937, 'weight_decay': 0.00058, 
                                                                    'box': 7.5, 'cls': 0.47651, 'dfl': 1.463,
                                                                    'degrees': 44.82499, 'scale': 0.4757, 'fliplr': 0.58144, 'flipud': 0.58409,
                                                                    'hsv_h': 0.01499, 'hsv_s': 0.7, 'hsv_v': 0.30742,
                                                                    },
                                                        'pretrained_model': 'YOLO_poland_10percentbkgd',
                                                        },
                     'pe_mckellar_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_mckellar'],
                                                   'source': 'global_birds_mckellar',
                                                   'default_loss_gains': {'domainclassifier': 1.0, 'unsuperviseddomainclassifier': 1.0,
                                                                            'multidomainclassifier': [1.0,1.0,1.0], 'unsupervisedmultidomainclassifier': [1.0,1.0,1.0],
                                                                            'featuresdistance': 0.25, 'unsupervisedfeaturesdistance': 0.25,
                                                                            'multifeaturesDC': 1.5, 'unsupervisedmultifeatsDC': 1.5
                                                                            },
                                                        'param': {'lr': 0.01107, 'lrf': 0.01096, 'momentum': 0.98, 'weight_decay': 0.00049, 
                                                                    'box': 6.84666, 'cls': 0.57734, 'dfl': 1.35745,
                                                                    'degrees': 34.17275, 'scale': 0.39214, 'fliplr': 0.44461, 'flipud': 0.38164,
                                                                    'hsv_h': 0.0141, 'hsv_s': 0.70406, 'hsv_v': 0.35903,
                                                                    },
                                                        'pretrained_model': '',
                                                    },
                      'pe_mckellar_10percentbkgd': {'datasets': ['global_birds_penguins', 'global_birds_mckellar'],
                                                   'source': 'global_birds_mckellar',
                                                   'default_loss_gains': {'domainclassifier': 1.0, 'unsuperviseddomainclassifier': 1.0,
                                                                            'multidomainclassifier': [1.0,1.0,1.0], 'unsupervisedmultidomainclassifier': [1.0,1.0,1.0],
                                                                            'featuresdistance': 0.25, 'unsupervisedfeaturesdistance': 0.25,
                                                                            'multifeaturesDC': 1.5, 'unsupervisedmultifeatsDC': 1.5
                                                                            },
                                                        'param': {'lr': 0.01107, 'lrf': 0.01096, 'momentum': 0.98, 'weight_decay': 0.00049, 
                                                                    'box': 6.84666, 'cls': 0.57734, 'dfl': 1.35745,
                                                                    'degrees': 34.17275, 'scale': 0.39214, 'fliplr': 0.44461, 'flipud': 0.38164,
                                                                    'hsv_h': 0.0141, 'hsv_s': 0.70406, 'hsv_v': 0.35903,
                                                                    },
                                                        'pretrained_model': 'YOLO_pe_10percent_background',
                                                    },
                        }


# For predictions
MATCH_IOU_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.3
CONF_THRESHOLD = 0.1

# For evaluation
NB_CONF_THRESHOLDS = 20
CONF_THRESHOLDS = np.linspace(0, 1, NB_CONF_THRESHOLDS) # CAREFUL: if you change that, don't forget to change calls to plot_confusion_matrix function


EXAMPLES_IMG = {
    'global_birds_palmyra': ['global_birds_palmyra_Dudley_projected_503_patch_760.0_0.0_640_640.jpg', 'global_birds_palmyra_Dudley_projected_559_patch_380.0_760.0_640_640.jpg', 'global_birds_palmyra_Dudley_projected_515_patch_0.0_760.0_640_640.jpg', 'global_birds_palmyra_Dudley_projected_736_patch_380.0_380.0_640_640.jpg', 'global_birds_palmyra_Dudley_projected_447_patch_760.0_760.0_640_640.jpg', 'global_birds_palmyra_Dudley_projected_559_patch_0.0_760.0_640_640.jpg', 'global_birds_palmyra_Dudley_projected_656_patch_380.0_380.0_640_640.jpg','global_birds_palmyra_Dudley_projected_656_patch_760.0_0.0_640_640.jpg', 'global_birds_palmyra_Dudley_projected_600_patch_380.0_0.0_640_640.jpg'],
    'global_birds_penguins': ['global_birds_penguins_cape_wallace_survey_8_481_patch_20.0_0.0_480_480.jpg', 'global_birds_penguins_cape_wallace_survey_8_481_patch_20.0_20.0_480_480.jpg', 'global_birds_penguins_cape_wallace_survey_8_633_patch_20.0_20.0_480_480.jpg', 'global_birds_penguins_cape_wallace_survey_8_542_patch_20.0_0.0_480_480.jpg', 'global_birds_penguins_cape_wallace_survey_8_604_patch_20.0_0.0_480_480.jpg', 'global_birds_penguins_cape_wallace_survey_8_602_patch_0.0_0.0_480_480.jpg'],
    'global_birds_mckellar': ['global_birds_mckellar_JackfishLakeBLTE_Sony_1_406_patch_0.0_0.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_448_patch_0.0_260.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_341_patch_260.0_260.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_874_patch_0.0_0.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_137_patch_260.0_260.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_341_patch_260.0_260.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_366_patch_260.0_260.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_369_patch_0.0_260.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_373_patch_0.0_260.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_874_patch_0.0_0.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_339_patch_260.0_0.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_414_patch_260.0_0.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_280_patch_260.0_0.0_640_640.jpg', 'global_birds_mckellar_JackfishLakeBLTE_Sony_1_414_patch_260.0_0.0_640_640.jpg'],
    'global_birds_pfeifer': ['global_birds_pfeifer_Fregata_Island_2016_Chinstrap_penguins_88_patch_2.0_2.0_448_448.jpg', 'global_birds_pfeifer_Fregata_Island_2016_Chinstrap_penguins_274_patch_0.0_0.0_448_448.jpg']
}