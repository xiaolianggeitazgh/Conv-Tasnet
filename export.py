
'''
##############evaluate trained models#################
python export.py
'''
import argparse
import numpy as np
from mindspore.train.serialization import export
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model import ConvTasNet

parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/tr',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default='/home/heu_MEDAI/RenQQ/The last/src/out/cv',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,   #取音频的长度，2s。#数据集语音长度要相同
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')  #最大音频长度，防止溢出
# Network architecture
parser.add_argument('--N', default=512, type=int,#256
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=20, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=256, type=int,#256
                    help='Number of channels in bottleneck 1 × 1-conv block')  #1 × 1-conv的通道数
parser.add_argument('--H', default=512, type=int,#512
                    help='Number of channels in convolutional blocks')   #卷几块的通道数
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')        #卷积核的大小
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat') #每次8个卷几块
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats') # 重复4次
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers') #说话者数量
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')  #归一化的方法
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')  #因果设定
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')  #产生mask的非线性层
# Training config
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU') #是否使用GPU，0==no
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs') #自动停止，如果10个epoch没有提升
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-6, type=float,
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
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')

def export_ConvTasnet():
    """ export """
    args = parser.parse_args()
    net = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                     args.C, norm_type=args.norm_type, causal=args.causal,
                     mask_nonlinear=args.mask_nonlinear)
    param_dict = load_checkpoint("/home/heu_MEDAI/RenQQ/5.24/src/exp/temp/ConvTasnet_4-10_3316.ckpt")
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.random.uniform(0.0, 1.0, size=[2, 32000]).astype(np.float32))
    export(net, input_data, file_name='ConvTasnet', file_format='MINDIR')
    print("export success")

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=2)
    export_ConvTasnet()
