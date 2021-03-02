import argparse
import kws.models

def set_params():
    params = {
        # The most important parameter
        'random_seed': 112233,

        # System params
        'verbose': True,
        'num_workers': 8,

        "comment": '', #help='comment in tensorboard title')
        # Wandb params
        'use_wandb': True,
        'wandb_project': 'google_speech_command',

        # Data location
        'train_dataset':'/Users/sampsonliu/Desktop/KWS/KWSCode/speech_commands/train',
        'valid_dataset':'/Users/sampsonliu/Desktop/KWS/KWSCode/speech_commands/valid',
        'background_noise':'/Users/sampsonliu/Desktop/KWS/KWSCode/speech_commands/train/_background_noise_',
        # 'data_root': '/Users/sampsonliu/Desktop/KWS/KWSCode/speech_commands', #'speech_commands/',
        # 'example_audio': 'example.wav',
        # 'example_fig': 'example_probs.jpg',

        # Checkpoints
        'checkpoint_dir': 'checkpoints/',
        'checkpoint_template': 'checkpoints/treasure_net{}.pt',
        'model_checkpoint': 'treasure_net.pt',
        'load_model': False,

        # Data processing
        'valid_ratio': 0.2,
        'fa_per_hour': 1.0,
        'audio_seconds': 1.0,
        'sample_rate': 16000,
        'time_steps': 81,
        'num_mels': 32,
        'keywords': [ 'yes', 'no', 'up', 'down', 'left','right', 'on', 'off', 'stop' , 'go'] ,#['marvin', 'sheila'],

        # Augmentation params:
        'mixup': False,
        'pitch_shift': 2.0,
        'noise_scale': 0.005,
        'gain_db': (-10.0, 30.0),
        'audio_scale': 0.15,


        #neural network models
        #choices=kws.models.available_models,
        #default=kws.models.available_models[0],
        'model': kws.models.available_models[0],

        # Optimizer params:
        'learning_rate': 1e-4,
        'lr_scheduler': 'plateau',# choices=['plateau', 'step'], help='method to adjust learning rate')
        'lr_scheduler_patience':5, #help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced'
        'lr_scheduler_step_size':50, #help='lr scheduler step: number of epochs of learning rate decay.'
        'lr_scheduler_gamma':0.1, #help='learning rate is multiplied by the gamma to decrease it'
        'weight_decay': 1e-2,
        'batch_size': 512,
        'optim':'sgd',# choices=['sgd', 'adam'],help='choices of optimization algorithms')
        'start_epoch': 1,
        'max_epochs': 100,

    }

    return params


