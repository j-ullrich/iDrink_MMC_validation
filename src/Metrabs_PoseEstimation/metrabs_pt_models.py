"""Loading of the Metrabs PyTorch models, shared by the 2D and 3D wrapper."""
import sys, os
import numpy as np
import simplepyutils as spu
import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'metrabs_pytorch'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'posepile'))
import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
import posepile.joint_info
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config

def load_multiperson_model(model_path):
    model_pytorch = load_crop_model(model_path)
    skeleton_infos = spu.load_pickle(f'{model_path}/skeleton_infos.pkl')
    joint_transform_matrix = np.load(f'{model_path}/joint_transform_matrix.npy')

    with torch.device('cuda'):
        return multiperson_model.Pose3dEstimator(
            model_pytorch.cuda(), skeleton_infos, joint_transform_matrix)


_model_cache = {}


def get_model(model_path):
    """
    Returns the multiperson model for model_path, loading it only once per process.

    Only one model is kept. Asking for another path evicts the old one and frees its GPU memory.
    """
    key = os.path.realpath(model_path)
    if key not in _model_cache:
        if _model_cache:
            _model_cache.clear()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        _model_cache[key] = load_multiperson_model(model_path).cuda()
    return _model_cache[key]


def load_crop_model(model_path):
    cfg = get_config()
    ji_np = np.load(f'{model_path}/joint_info.npz')
    ji = posepile.joint_info.JointInfo(ji_np['joint_names'], ji_np['joint_edges'])
    backbone_raw = getattr(effnet_pt, f'efficientnet_v2_{cfg.efficientnet_size}')()
    preproc_layer = effnet_pt.PreprocLayer()
    backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
    model = metrabs_pt.Metrabs(backbone, ji)
    model.eval()

    inp = torch.zeros((1, 3, cfg.proc_side, cfg.proc_side), dtype=torch.float32)
    intr = torch.eye(3, dtype=torch.float32)[np.newaxis]

    model((inp, intr))
    model.load_state_dict(torch.load(f'{model_path}/ckpt.pt'))
    return model
