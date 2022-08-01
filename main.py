from utils_lib.utils import Utils

from time import process_time
import argparse
from utils_lib.train_master import TrainMaster

parser = argparse.ArgumentParser(description='Benchmark all params')

parser.add_argument('--mode', default="train",
                    help='Mode train/manual/plot/show (default: train)')

parser.add_argument('--multi', action='store_true')
parser.add_argument('--no-multi', dest='multi', action='store_false')
parser.set_defaults(multi=False)

parser.add_argument('--env', type=int, default=0, metavar='N',
                    help='')
parser.add_argument('--policy', type=int, default=0, metavar='N',
                    help='')
parser.add_argument('--nb-timesteps', type=int, default=0, metavar='N',
                    help='')
parser.add_argument('--etp', default="r",
                    help='Element to plot, c,r,t (default: r)')
parser.add_argument('--feature', type=int, default=0, metavar='N',
                    help='')

parser.add_argument('--index', type=int, default=0, metavar='N',
                    help='')


# parser.add_argument('--tau', type=float, default=0.005, metavar='G',
#                     help='target smoothing coefficient(τ) (default: 0.005)')

args = parser.parse_args()
trainer = TrainMaster()




#python main.py --mode train --multi True --index 0
if args.mode == 'train':
    print(args)
    if args.multi:
        trainer.train_all_envs(
            offset_env=args.env,
            offset_policie=args.policy,
            index=args.index
        )
    else:
        trainer.train_env_with_all_tunning(
            env_j=args.env,
            offset_policie=args.policy,
            index=args.index,
        )
if args.mode == 'manual':
    print(args)
    
    trainer.train_and_bench(
        policie_i=args.policy,
        env_j=args.env,
        fe_k=args.feature,
        index=args.index,
        #nb_train=120
        )
    with open("multi_exe.log","a") as f:
        f.write(str(args.env) + " " + str(args.policy) + " " + str(args.feature) + " " + str(args.index))
        f.write("\n")
        f.close()





if args.mode == 'show':
    
    trainer.show_env_policie_fe(
        policie_i=args.policy,
        env_j=args.env,
        fe_k=0,
        fev_l=0
        )
 