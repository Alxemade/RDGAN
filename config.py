from easydict import EasyDict as edict
import json
import os


config = edict()
config.TRAIN = edict()

config.TRAIN.batch_size = 25
config.TRAIN.early_stopping_num = 10
config.TRAIN.lr = 0.0001
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 5
config.TRAIN.beta1 = 0.5
config.TRAIN.beta2 = 0.9
config.TRAIN.n_epoch = 50
config.TRAIN.sample_size = 50
config.TRAIN.g_alpha = 15
config.TRAIN.g_gamma = 2e-5
config.TRAIN.g_beta = 0.1
config.TRAIN.g_adv = 1
config.TRAIN.layer_norm = False
config.TRAIN.seed = 100
config.TRAIN.epsilon = 0.000001
config.TRAIN.n_critic = 5
config.TRAIN.VGG19_path = os.path.join('trained_model', 'VGG19', 'vgg19.npy')
config.TRAIN.training_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'training.pickle')
config.TRAIN.val_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'validation.pickle')
config.TRAIN.testing_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'testing.pickle')
config.TRAIN.evaluate_data_sample_path = os.path.join('evaluate', 'samples')
config.TRAIN.evaluate_data_save_path = os.path.join('evaluate', 'save')
config.TRAIN.mask_RadialApproxOnCartesi_path = os.path.join('mask', 'RadialApproxOnCartesi')
config.TRAIN.mask_Gaussian1D_path = os.path.join('mask', 'Gaussian1D')
config.TRAIN.mask_Gaussian2D_path = os.path.join('mask', 'Gaussian2D')
config.TRAIN.mask_Poisson2D_path = os.path.join('mask', 'Poisson2D')


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")

