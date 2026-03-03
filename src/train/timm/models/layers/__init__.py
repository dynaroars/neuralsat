# NOTE ..models.layers is DEPRECATED, please use ..layers, this is here to reduce breakages in transition
from ..layers.activations import *
from ..layers.adaptive_avgmax_pool import \
    adaptive_avgmax_pool2d, select_adaptive_pool2d, AdaptiveAvgMaxPool2d, SelectAdaptivePool2d
from ..layers.attention_pool2d import AttentionPool2d, RotAttentionPool2d, RotaryEmbedding
from ..layers.blur_pool import BlurPool2d
from ..layers.classifier import ClassifierHead, create_classifier
from ..layers.cond_conv2d import CondConv2d, get_condconv_initializer
from ..layers.config import is_exportable, is_scriptable, is_no_jit, set_exportable, set_scriptable, set_no_jit,\
    set_layer_config
from ..layers.conv2d_same import Conv2dSame, conv2d_same
from ..layers.conv_bn_act import ConvNormAct, ConvNormActAa, ConvBnAct
from ..layers.create_act import create_act_layer, get_act_layer, get_act_fn
from ..layers.create_attn import get_attn, create_attn
from ..layers.create_conv2d import create_conv2d
from ..layers.create_norm import get_norm_layer, create_norm_layer
from ..layers.create_norm_act import get_norm_act_layer, create_norm_act_layer, get_norm_act_layer
from ..layers.drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from ..layers.eca import EcaModule, CecaModule, EfficientChannelAttn, CircularEfficientChannelAttn
from ..layers.evo_norm import EvoNorm2dB0, EvoNorm2dB1, EvoNorm2dB2,\
    EvoNorm2dS0, EvoNorm2dS0a, EvoNorm2dS1, EvoNorm2dS1a, EvoNorm2dS2, EvoNorm2dS2a
from ..layers.fast_norm import is_fast_norm, set_fast_norm, fast_group_norm, fast_layer_norm
from ..layers.filter_response_norm import FilterResponseNormTlu2d, FilterResponseNormAct2d
from ..layers.gather_excite import GatherExcite
from ..layers.global_context import GlobalContext
from ..layers.helpers import to_ntuple, to_2tuple, to_3tuple, to_4tuple, make_divisible, extend_tuple
from ..layers.inplace_abn import InplaceAbn
from ..layers.linear import Linear
from ..layers.mixed_conv2d import MixedConv2d
from ..layers.mlp import Mlp, GluMlp, GatedMlp, ConvMlp
from ..layers.non_local_attn import NonLocalAttn, BatNonLocalAttn
from ..layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d
from ..layers.norm_act import BatchNormAct2d, GroupNormAct, convert_sync_batchnorm
from ..layers.padding import get_padding, get_same_padding, pad_same
from ..layers.patch_embed import PatchEmbed
from ..layers.pool2d_same import AvgPool2dSame, create_pool2d
from ..layers.squeeze_excite import SEModule, SqueezeExcite, EffectiveSEModule, EffectiveSqueezeExcite
from ..layers.selective_kernel import SelectiveKernel
from ..layers.separable_conv import SeparableConv2d, SeparableConvNormAct
from ..layers.split_attn import SplitAttn
from ..layers.split_batchnorm import SplitBatchNorm2d, convert_splitbn_model
from ..layers.std_conv import StdConv2d, StdConv2dSame, ScaledStdConv2d, ScaledStdConv2dSame
from ..layers.test_time_pool import TestTimePoolHead, apply_test_time_pool
from ..layers.trace_utils import _assert, _float_to_int
from ..layers.weight_init import trunc_normal_, trunc_normal_tf_, variance_scaling_, lecun_normal_

import warnings
warnings.warn(f"Importing from {__name__} is deprecated, please import via ..layers", DeprecationWarning)
