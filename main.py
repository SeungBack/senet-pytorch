import argparse
from train import trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # training hyperparameter
    parser.add_argument("-e","--max_epoch", type=int, default=200, help="number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("-bs","--batch_size", type=int, default=128, help="batch size")

    # data loader
    parser.add_argument("--num_workers", type=int, default=6, help="number of workers for data loader")

    # model setting
    parser.add_argument("--block", type=str, default='res', help="res, se, mse, wse")
    parser.add_argument("--inplanes", type=int, default=64, help="number of in planes of resnet.")
    parser.add_argument("--load_weights", type=str, default=None, help="path to the pretrained weights")

    # inference
    parser.add_argument("--input_folder", type=str, default='./datasets/cropped_sample', help="path to the dataset root")
    parser.add_argument("--output_folder", type=str, default='./results', help="path to the dataset root")

    args = parser.parse_args()

    trainer(args)