from utils_lib.utils import Utils

from time import process_time
import argparse
from utils_lib.train_master import TrainMaster

parser = argparse.ArgumentParser(description='Benchmark all params')

parser.add_argument('--mode', default="manual",
                    help='Mode train/manual/manual_all (default: train)')

parser.add_argument('--compute', default="auto",
                    help='Mode cpu/auto (default: auto)')


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
parser.add_argument('--feature_var', type=int, default=0, metavar='N',
                    help='')

parser.add_argument('--index', type=int, default=0, metavar='N',
                    help='')

args = parser.parse_args()
trainer = TrainMaster(device=args.compute)

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
    trainer.train_and_bench_all_fev(
        policie_i=args.policy,
        env_j=args.env,
        fe_k=args.feature,
        index=args.index,
        #nb_train=120
        )
    print("end")

if args.mode == 'manual_all':
    trainer.train_and_bench(
        policie_i=args.policy,
        env_j=args.env,
        fe_k = args.feature,
        fev_l=args.feature_var,
        index=args.index,
    )


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
# if args.mode == 'manual':
#     print(args)
#     trainer.train_and_bench_all_fev(
#         policie_i=args.policy,
#         env_j=args.env,
#         fe_k=args.feature,
#         index=args.index,
#         #nb_train=120
#         )

if args.mode == 'create_folder':
    print("ok")
    u = Utils()
    u.init_folder()
    print("create_folder")
