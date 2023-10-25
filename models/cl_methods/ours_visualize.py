import torch.nn as nn
from models import CrossModalAttention


class Ours_Visualize(nn.Module):

    def __init__(self, model: nn.Module, embedding_dim: int, device,
                 avm_pretrain_path: str, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        self.backbone.requires_grad_(requires_grad=False)

        # Audio-video matching module initialization
        self.avmatching_module = CrossModalAttention(
            dim=embedding_dim,
            pretrain_path=avm_pretrain_path,
        )
        # Buffer initialization
        self.device = device

        self.num_freq_tokens = 8

        self._req_penalty = False
        self._req_opt = False

    def forward(self, inputs):
        if 'return_attn' in inputs and inputs['return_attn']:  # For visualization
            return self.return_attention(inputs)
        else:
            raise ValueError("Only supports visualization here")


    def return_attention(self, inputs):

        output = self.backbone(inputs)

        video_embeds = output['inter_c_v']
        audio_embeds = output['inter_c_a']

        pos_code_inputs = {
            "video_data": video_embeds,
            "audio_data": audio_embeds,
            "audio_code_inputs": True,
            "video_code_inputs": True,
            "joint_token": True,
        }
        pos_output = self.backbone(pos_code_inputs)
        pos_cross_attn_av, pos_cross_attn_va = self.avmatching_module.infer_attention(pos_output['embedding_a'],
                                                                      pos_output['embedding_v'],
                                                                      compute_av_positive=True,
                                                                      compute_va_positive=True,
                                                                        normalize=True)
        pos_cross_attn_av = pos_cross_attn_av.mean(dim=1).mean(dim=1)
        pos_cross_attn_va = pos_cross_attn_va.mean(dim=1).mean(dim=1)

        return pos_cross_attn_av, pos_cross_attn_va