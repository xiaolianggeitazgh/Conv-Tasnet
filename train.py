
import argparse
import os
from mindspore import Model
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import nn
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
import mindspore.dataset as ds
from src.data import DatasetGenerator
from src.network_define import WithLossCell
from src.Loss import loss
from src.model import ConvTasNet
from src.preprocess import preprocess


parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
parser.add_argument('--train_dir', type=str, default="/mass_data/dataset/LS-2mix/Libri2Mix/tt/",
                    help='directory including mix.json, s1.json and s2.json')

parser.add_argument('--valid_dir', type=str, default='/home/heu_MEDAI/RenQQ/The last/src/out/cv',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
parser.add_argument('--N', default=512, type=int,#256
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=20, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=256, type=int,#256
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,#512
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
# Training config
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.01, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument("--run_distribute", type=int, default=0,
                    help="run distribute, default: false.")
parser.add_argument("--device_id", type=int, default=6,
                    help="device id, default: 0.")
parser.add_argument("--device_num", type=int, default=1,
                    help="number of device, default: 0.")
parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'),
                    help='device where the code will be implemented (default: Ascend)')

parser.add_argument('--data_url', default='/home/work/user-job-dir/inputs/data/',
                    help='path to training/inference dataset folder')
parser.add_argument('--train_url', default='/home/work/user-job-dir/model/',
                    help='model folder to save/load')
parser.add_argument('--in_dir', type=str, default=r"/home/work/user-job-dir/inputs/data/",
                    help='Directory path of wsj0 including tr, cv and tt')
parser.add_argument('--out_dir', type=str, default=r"/home/work/user-job-dir/inputs/data_json",
                    help='Directory path to put output files')
parser.add_argument('--modelArts', default=0, type=int,
                    help='Continue from checkpoint model')
parser.add_argument('--continue_train', default=0, type=int,
                    help='Continue from checkpoint model')
parser.add_argument('--ckpt_path', type=str, default="DPTNet-10_890.ckpt",
                    help='Path to model file created by training')

def main(args):
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    device_num = int(os.environ.get("RANK_SIZE", 1))
    if device_num == 1:
        is_distributed = 'False'
    elif device_num > 1:
        is_distributed = 'True'

    if is_distributed == 'True':
        print("parallel init", flush=True)
        init()
        rank_id = get_rank()
        context.reset_auto_parallel_context()
        parallel_mode = ParallelMode.DATA_PARALLEL
        rank_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=args.device_num)
        context.set_auto_parallel_context(parameter_broadcast=True)
        print("Starting traning on multiple devices...")
    else:
        if args.modelArts:
            init()
            rank_id = get_rank()
            rank_size = get_group_size()
        else:
            context.set_context(device_id=args.device_id)
        if args.modelArts:
            import moxing as mox
            obs_data_url = args.data_url
            args.data_url = '/home/work/user-job-dir/inputs/data/'
            obs_train_url = args.train_url

            home = os.path.dirname(os.path.realpath(__file__))
            train_dir = os.path.join(home, 'checkpoints') + str(rank_id)
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)

            save_checkpoint_path = train_dir + '/device_' + os.getenv('DEVICE_ID') + '/'
            if not os.path.exists(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)
            save_ckpt = os.path.join(save_checkpoint_path, 'dptnet.ckpt')

            mox.file.copy_parallel(obs_data_url, args.data_url)
            print("Successfully Download {} to {}".format(obs_data_url, args.data_url))

            print("start preprocess on modelArts....")
            preprocess(args)

    print("Start datasetgenerator")
    tr_dataset = DatasetGenerator(args.train_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)

    print("start Generatordataset")
    if is_distributed == 'True':
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"],
                                        shuffle=False, num_shards=rank_size, shard_id=rank_id)
    else:
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"],
                                        shuffle=False)
    tr_loader = tr_loader.batch(1)

    print("data loading done")

    net = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                     args.C, norm_type=args.norm_type, causal=args.causal,
                     mask_nonlinear=args.mask_nonlinear)

    if args.continue_train:
        if args.modelArts:
            home = os.path.dirname(os.path.realpath(__file__))
            ckpt = os.path.join(home, args.ckpt_path)
            params = load_checkpoint(ckpt)
            load_param_into_net(net, params)
        else:
            params = load_checkpoint(args.ckpt_path)
            load_param_into_net(net, params)
    print(net)
    net.set_train()
    optimizer = nn.SGD(net.trainable_params(), learning_rate=args.lr, weight_decay=args.l2, momentum=args.momentum)

    my_loss = loss()
    net_with_loss = WithLossCell(net, my_loss)
    model = Model(net_with_loss, optimizer=optimizer)

    time_cb = TimeMonitor()
    loss_cb = LossMonitor(1)
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=1,
                                 keep_checkpoint_max=1)
    if args.modelArts:
        ckpt_cb = ModelCheckpoint(prefix="Conv-TasNet", directory=save_ckpt, config=config_ck)
    else:
        ckpt_cb = ModelCheckpoint(prefix="Conv-TasNet", directory=args.save_folder, config=config_ck)
    cb += [ckpt_cb]
    model.train(epoch=args.epochs, train_dataset=tr_loader, callbacks=cb, dataset_sink_mode=True)
    if args.modelArts:
        import moxing as mox
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir, obs_train_url))


if __name__ == '__main__':
    arg = parser.parse_args()
    print(arg)
    main(arg)
