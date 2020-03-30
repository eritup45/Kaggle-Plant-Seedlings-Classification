import argparse

def parse_args():
    desc = 'PyTorch example code for Kaggle competition -- Plant Seedlings Classification.\n' \
           'See https://www.kaggle.com/c/plant-seedlings-classification'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-p', '--path', help='path to dataset', type=str, default="./")
    parser.add_argument('--pretrained_model_path', help='path to pretrained model', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=30)                   # number of epochs, default: 50
    parser.add_argument('--batch_size', type=int, default=32)               # batch size, default: 32
    parser.add_argument('--lr', type=float, default=0.001)                  # learning rate, default:0.001, 因為用 Adam 所以建議設小一點
    parser.add_argument('--num_workers', type=int, default=4)               # number of workers, default: 4
    parser.add_argument('--cuda', action='store_true', default=True)    # disable cuda device, default: False
    return parser.parse_args()
