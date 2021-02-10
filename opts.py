import argparse
import models

def parse_opt():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #Training Dataset
    parser.add_argument(
        "--train-dataset",
        type=str,
        default='/Users/sampsonliu/Desktop/KWS/KWSCode/speech_commands/train',
        help='path of train dataset')
    # Valiation Dataset
    parser.add_argument(
        "--valid-dataset",
        type=str,
        default='/Users/sampsonliu/Desktop/KWS/KWSCode/speech_commands/valid',
        help='path of validation dataset')

    parser.add_argument(
        "--background-noise",
        type=str,
        default='/Users/sampsonliu/Desktop/KWS/KWSCode/speech_commands/train/_background_noise_',
        help='path of background noise')

    parser.add_argument(
        "--comment",
        type=str,
        default='',
        help='comment in tensorboard title')

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help='batch size')

    parser.add_argument(
        "--dataload-workers-nums",
        type=int,
        default=6,
        help='number of workers for dataloader')

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help='weight decay')

    parser.add_argument(
        "--optim",
        choices=['sgd', 'adam'],
        default='sgd',
        help='choices of optimization algorithms')

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help='learning rate for optimization')

    parser.add_argument(
        "--lr-scheduler",
        choices=['plateau', 'step'],
        default='plateau',
        help='method to adjust learning rate')

    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=5,
        help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')


    parser.add_argument(
        "--lr-scheduler-step-size",
        type=int,
        default=50,
        help='lr scheduler step: number of epochs of learning rate decay.')

    parser.add_argument(
        "--lr-scheduler-gamma",
        type=float,
        default=0.1,
        help='learning rate is multiplied by the gamma to decrease it')

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=70,
        help='max number of epochs')

    parser.add_argument(
        "--resume",
        type=str,
        help='checkpoint file to resume')

    parser.add_argument(
        "--model",
        choices=models.available_models,
        default=models.available_models[10],
        help='model of NN')

    parser.add_argument(
        "--input",
        choices=['mel32', 'mel40'],
        default='mel32',
        help='input of NN')

    parser.add_argument(
        '--mixup',
        action='store_true',
        help='use mixup')


    args = parser.parse_args()

    return args
