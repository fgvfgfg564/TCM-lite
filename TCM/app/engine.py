import torch
import torch.nn as nn
import os

from ..models.tcm import *
from ..tensorrt_support import *

@tensorrt_compiled_module
class G_a(nn.Sequential):
    def __init__(self, head_dim, window_size, dim, dpr, begin, config, N, M, block_size):
        m_down1 = [ConvTransBlock(dim, dim, head_dim[0], window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW')
                        for i in range(config[0])] + \
                       [ResidualBlockWithStride(2 * N, 2 * N, stride=2)]
        m_down2 = [ConvTransBlock(dim, dim, head_dim[1], window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW')
                        for i in range(config[1])] + \
                       [ResidualBlockWithStride(2 * N, 2 * N, stride=2)]
        m_down3 = [ConvTransBlock(dim, dim, head_dim[2], window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW')
                        for i in range(config[2])] + \
                       [conv3x3(2 * N, M, stride=2)]
        super().__init__(*[ResidualBlockWithStride(3, 2 * N, 2)] + m_down1 + m_down2 + m_down3)
        self.input_shape = [1, 3, block_size, block_size]

@tensorrt_compiled_module
class G_s(nn.Sequential):
    def __init__(self, head_dim, window_size, dim, dpr, begin, config, N, M, block_size):
        m_up1 = [ConvTransBlock(dim, dim, head_dim[3], window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW')
                      for i in range(config[3])] + \
                     [ResidualBlockUpsample(2 * N, 2 * N, 2)]
        m_up2 = [ConvTransBlock(dim, dim, head_dim[4], window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW')
                      for i in range(config[4])] + \
                     [ResidualBlockUpsample(2 * N, 2 * N, 2)]
        m_up3 = [ConvTransBlock(dim, dim, head_dim[5], window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW')
                      for i in range(config[5])] + \
                     [subpel_conv3x3(2 * N, 3, 2)]
        super().__init__(*[ResidualBlockUpsample(M, 2 * N, 2)] + m_up1 + m_up2 + m_up3)
        self.input_shape = [1, M, block_size // 16, block_size // 16]

class TCMModelEngine(CompressionModel):
    def __init__(self, mode, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=128, M=320, num_slices=5, max_support_slices=5, block_size=512, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        if mode == 'encode':
            # The encoder-only modules
            self.g_a = G_a(self.head_dim, self.window_size, dim, dpr, begin, config, N, M, block_size)
            self.ha_down1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i % 2 else 'SW')
                            for i in range(config[0])] + \
                            [conv3x3(2 * N, 192, stride=2)]

            self.h_a = nn.Sequential(
                *[ResidualBlockWithStride(320, 2 * N, 2)] + \
                self.ha_down1
            )

        elif mode == 'decode':
            # The decoder-only modules
            self.g_s = G_s(self.head_dim, self.window_size, dim, dpr, begin, config, N, M, block_size)

        # The shared modules
        self.hs_up1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i % 2 else 'SW')
                       for i in range(config[3])] + \
                      [subpel_conv3x3(2 * N, 320, 2)]

        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2 * N, 2)] + \
             self.hs_up1
        )

        self.hs_up2 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i % 2 else 'SW')
                       for i in range(config[3])] + \
                      [subpel_conv3x3(2 * N, 320, 2)]

        self.h_scale_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2 * N, 2)] + \
             self.hs_up2
        )

        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320 // self.num_slices) * min(i, 5)), (320 + (320 // self.num_slices) * min(i, 5)), 16,
                        self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320 // self.num_slices) * min(i, 5)), (320 + (320 // self.num_slices) * min(i, 5)), 16,
                        self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320 // self.num_slices) * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320 // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320 // self.num_slices) * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320 // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320 // self.num_slices) * min(i + 1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320 // self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

        self.modnet = Modnet(M, N)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x, lmd=0):
        b = x.size()[0]

        if lmd == 0:  # training
            rand_lambda = np.random.rand(b) 
            rand_lambda = 0.002 * rand_lambda + 0.001
            lmd_info = np.array(rand_lambda, dtype=np.float32)

            lmd_info = torch.from_numpy(lmd_info).cuda()
            lmd_info = torch.reshape(lmd_info, (b, 1)).cuda()
        else:
            lmd_info = lmd * torch.ones((b, 1))
            lmd_info = lmd_info.cuda()

        y = self.g_a(x)
        y = self.modnet(y, lmd_info)

        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu
            # if self.training:
            #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
            # else:
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "lmd_info": lmd_info,
        }

    def load_state_dict(self, state_dict, strict=True):
        # todo: change
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        nn.Module.load_state_dict(self, state_dict, strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[1][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
    
    def compile(self, target_path):
        compile(self, target_path)
    
    def load(self, weight_path):
        if os.path.isdir(weight_path):
            load_weights(self, weight_path)
        else:
            # Not compiled yet; a checkpoint from training
            state_dict = torch.load(weight_path)
            self.load_state_dict(state_dict["state_dict"], strict=False)