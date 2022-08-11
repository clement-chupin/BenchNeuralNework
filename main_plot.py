from utils_lib.utils import Utils

from time import process_time
import argparse
from utils_lib.train_master import TrainMaster

parser = argparse.ArgumentParser(description='Benchmark all params')

parser.add_argument('--mode', default="train",
                    help='Mode train/manual/manual_all (default: train)')

args = parser.parse_args()

if args.mode == 'train':
    print(args)



