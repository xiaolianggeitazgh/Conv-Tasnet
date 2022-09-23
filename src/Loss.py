
from itertools import permutations
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import nn
from mindspore import Tensor

EPS = 1e-8

class loss(nn.Cell):
    def __init__(self):
        super(loss, self).__init__()
        self.mean = ops.ReduceMean()
        self.cast = ops.Cast()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.expand_dims = ops.ExpandDims()
        self._sum = ops.ReduceSum(keep_dims=False)
        self.log = ops.Log()
        self.scatter = ops.ScatterNd()
        self.matmul = ops.MatMul()
        self.transpose = ops.Transpose()
        self.Argmax = ops.Argmax(axis=1, output_type=mindspore.int32)
        self.argmax = ops.ArgMaxWithValue(axis=1, keep_dims=True)
        self.ones = ops.Ones()
        self.zeros_like = ops.ZerosLike()
        self.log10 = Tensor(np.array([10.0]), mindspore.float32)
        self.perms = Tensor(list(permutations(range(2))), dtype=mindspore.int64)
        self.perms_one_hot = Tensor(np.array([[1, 0], [0, 1], [0, 1], [1, 0]]), mindspore.float32)

        self.print = ops.Print()

    def construct(self, source, estimate_source, source_lengths):
        return self.cal_loss(source, estimate_source, source_lengths)

    def cal_loss(self, source, estimate_source, source_lengths):
        """
        Args:
            source: [B, C, T], B is batch size
            estimate_source: [B, C, T]
            source_lengths: [B]
        """
        max_snr, perms, max_snr_idx = self.cal_si_snr_with_pit(source, estimate_source, source_lengths)
        loss_cal = 0 - self.mean(max_snr)
        reorder_estimate_source = self.reorder_source(estimate_source, perms, max_snr_idx)
        return loss_cal, max_snr, estimate_source, reorder_estimate_source


    def cal_si_snr_with_pit(self, source, estimate_source, source_lengths):
        """Calculate SI-SNR with PIT training.
        Args:
            source: [B, C, T], B is batch size
            estimate_source: [B, C, T]
            source_lengths: [B], each item is between [0, T]
        """
        B, C, _ = source.shape
        # mask padding position along T
        mask = self.get_mask(source, source_lengths)
        estimate_source *= mask

        # Step 1. Zero-mean norm
        num_samples = self.cast(source_lengths.view(-1, 1, 1), mindspore.float32)  # [B, 1, 1]
        mean_target = self.sum(source, 2) / num_samples
        mean_estimate = self.sum(estimate_source, 2) / num_samples
        zero_mean_target = source - mean_target
        zero_mean_estimate = estimate_source - mean_estimate
        # mask padding position along T
        zero_mean_target *= mask
        zero_mean_estimate *= mask

        # Step 2. SI-SNR with PIT
        # reshape to use broadcast
        s_target = self.expand_dims(zero_mean_target, 1)  # [B, 1, C, T]
        s_estimate = self.expand_dims(zero_mean_estimate, 2)  # [B, C, 1, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = self.sum(s_estimate * s_target, 3)  # [B, C, C, 1]
        s_target_energy = self.sum(s_target ** 2, 3) + EPS  # [B, 1, C, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = self._sum(pair_wise_proj ** 2, 3) / (self._sum(e_noise ** 2, 3) + EPS)
        pair_wise_si_snr = 10 * self.log(pair_wise_si_snr + EPS) / self.log(
            self.log10)  # [B, C, C]

        # Get max_snr of each utterance
        # permutations, [C!, C]
        perms = self.perms
        # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
        perms_one_hot = self.perms_one_hot
        snr_set = self.matmul(pair_wise_si_snr.view(B, -1), perms_one_hot)
        max_snr_idx = self.Argmax(snr_set)  # [B]
        # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
        _, max_snr = self.argmax(snr_set)
        max_snr /= C
        return max_snr, perms, max_snr_idx

    def reorder_source(self, source, perms, max_snr_idx):
        """
        Args:
            source: [B, C, T]
            perms: [C!, C], permutations
            max_snr_idx: [B], each item is between [0, C!)
        Returns:
            reorder_source: [B, C, T]
        """
        B, C, _ = source.shape
        # [B, C], permutation whose SI-SNR is max of each utterance
        # for each utterance, reorder estimate source according this permutation
        max_snr_perm = perms[max_snr_idx, :]
        reorder_source = self.zeros_like(source)
        for b in range(B):
            for c in range(C):
                if max_snr_perm[b][c] == 1:
                    reorder_source[b, c] = source[b, 1]
                else:
                    reorder_source[b, c] = source[b, 0]
        return reorder_source


    def get_mask(self, source, source_lengths):
        """
        Args:
            source: [B, C, T]
            source_lengths: [B]
        Returns:
            mask: [B, 1, T]
        """
        B, _, T = source.shape
        mask = self.ones((B, 1, T), mindspore.float32)
        for i in range(B):
            mask[i, :, 46400:] = 0
        return mask

if __name__ == "__main__":
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=4)
    my_loss = loss()
    print("______________________ test cal_loss _______________________")
    padded_source = Tensor(np.random.randn(2, 2, 46400), dtype=mindspore.float32)
    mixture_lengths = Tensor([46400, 46400], dtype=mindspore.int32)
    estimate_source_test = Tensor(np.random.randn(2, 2, 46400), dtype=mindspore.float32)
    loss, max_snr_test, estimate_source_test, reorder_estimate_source_test = \
        my_loss(padded_source, estimate_source_test, mixture_lengths)
    print("#" * 100)
