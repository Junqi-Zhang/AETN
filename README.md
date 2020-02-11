# AETN
The implementation of the AutoEncoder-coupled Transformer Network by TensorFlow

## Introduction
The AutoEncoder-coupled Transformer Network is a novel model for general-purpose user embedding learning based on mobile app usage. This implementation is a TensorFlow version, supporting multi-GPUs training, and employs the tf.dataset and tf.estimator. You can train the AETN or the V-AETN with optional task of auxiliary retention reconstruction and masked app prediction by run the train.py. Models are defined in the './Module/models.py'.

## System requirements
* Python >= 3.6.5
* Numpy >= 1.14.3
* Pandas >= 0.23.0
* TensorFlow = 1.12
* Total memory of GPUs >= 16G

## Data preprocesses

## Settings in train.py
* Dataset Path
```Python
DATA_DIR = './'
SOFTMAP_DIR = DATA_DIR + 'softmap/'
SOFTMAP_PLUS_DIR = DATA_DIR + 'softmap2/'
TRAIN_DIR = DATA_DIR + 'train'
VAL_DIR = DATA_DIR + 'valid'
TEST_DIR = DATA_DIR + 'test'

NUM_TRAIN_SAMPLE = 1.6e+7 # Total Number of Samples in Training Dataset
LENGTH_HIS = 25 # The Length of Installation or Uninstallation
BERT_MASK_NUM = 3 # How Many Apps Will Be Masked in the Masked App Prediction
BATCH_SIZE = int(1000/NUM_GPUS) # Batch Size. Pay Attention to the NUM_GPU!!!

train_data_params = {
    'data_dir': TRAIN_DIR,
    'file_stamp': 'csv',
    'header': True, # Whether TRAIN_DIR.csv Has a Header
    'shuffle': True,
    '''
    ...
    '''
    'batch_size': BATCH_SIZE,
    'length_his': LENGTH_HIS,
    'bert_mask_num': BERT_MASK_NUM
}
```
* Model Settings
1. MODEL_TYPE = AETN or MODEL_TYPE = VAETN
2. If trained with the masked app prediction task, 'train_mlm' should be set to True.
3. If trained with the auxiliary retention reconstruction task, 'alpha_dae' should be set to 1.0.
```
model_params = {
    'model_type': MODEL_TYPE,
    'num_train_sample': NUM_TRAIN_SAMPLE,
    'batch_size': BATCH_SIZE,

    'train_mlm': True,
    'p_dims': [256, 512, NUM_SOFTID],
    'q_dims': None,
    'u_dims': 128,
    'l_dims': [1024, NUM_LABEL],
    'ffn_dim': 1024,
    'num_translayer': 2,
    'num_header': 8,
    'num_date': 52,
    'length_his': LENGTH_HIS,
    'softmap': SOFTMAP_PLUS['type_idx'].values,
    'softmap_vocab': len(SOFTMAP_PLUS[['type', 'type_idx']].drop_duplicates()),

    'image_dropout_rate': 0.05,
    'classifier_dropout_rate': 0.05,
    'attention_dropout_rate': 0.1,
    'ffn_dropout_rate': 0.1,

    'lambda_L2': 1.5e-7,

    'loss_type_dae': 'logistic',
    'weight_positive_dae': 100,
    'alpha_dae': 1.0,

    'loss_type_image': 'logistic',
    'weight_positive_image': 100,

    'loss_type_his': 'softmax',
    'weight_loss_his': 1.0,

    'alpha_main': 1.0,

    'loss_type_classifier': 'logistic',
    'weight_positive_classifier': 100,
    'alpha_classifier': 0,

    'optimizer': 'Adam',
    'init_lr': 1e-4,
    'decay_steps': 16000,
    'decay_rate': 0.8,

    'monitor_metric': 'loss_main',
}
```
