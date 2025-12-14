import os
import torch
from models.condition_models import ns_Transformer, PatchTST_TS, SVQ, iTransformer
# from models.diffusion_models import diffuMTS

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'ns_Transformer': ns_Transformer,
            'PatchTST':PatchTST_TS,
            'SVQ': SVQ,
            'iTransformer': iTransformer
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)


    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # CUDA_VISIBLE_DEVICES is already set in cond_model_main.py before any CUDA operations
            # So we always use cuda:0 which maps to the specified physical GPU
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0 (Physical GPU {} via CUDA_VISIBLE_DEVICES)'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
