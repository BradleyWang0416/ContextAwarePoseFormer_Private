import torch
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn import TransformerDecoderLayer
import torch.nn.functional as F
from einops import rearrange

from mvn.models import pose_hrnet
from mvn.models.networks import network
from mvn.models.cpn.test_config import cfg
from mvn.models.pose_dformer import PoseTransformer
from mvn.utils.viz_skel_seq import viz_skel_seq_anim

class CA_PF(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints

        if config.model.backbone.type in ['hrnet_32', 'hrnet_48']:
            self.backbone = pose_hrnet.get_pose_net(config.model.backbone)

        elif config.model.backbone.type == 'cpn':
            self.backbone = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=False)

        if config.model.backbone.fix_weights:
            print("model backbone weights are fixed")
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.volume_net = PoseTransformer(config.model.poseformer, backbone=config.model.backbone.type)

    def forward(self, images, keypoints_2d_cpn, keypoints_2d_cpn_crop): # [B,256,192,3], [B,17,2], [B,17,2]
        device = keypoints_2d_cpn.device
        images = images.permute(0, 3, 1, 2).contiguous()    # [B,256,192,3] -> [B,3,256,192]

        keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        # forward backbone
        features_list = self.backbone(images) 
        # features_list: List[Tensor]. len=4. example: [torch.Size([B,32,64,48]), torch.Size([B,64,32,24]), torch.Size([B,128,16,12]), torch.Size([B,256,8,6])]
        keypoints_3d = self.volume_net(keypoints_2d_cpn, keypoints_2d_cpn_crop, features_list)

        return keypoints_3d # [B,1,17,3]


class CA_PF_VIDEO(CA_PF):
    def __init__(self, *args, memory_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)

        

        self.num_feature_levels = memory_cfg.get('num_feature_levels', 4)   # 4 for hrnet
        self.num_maskmem = memory_cfg.get('num_maskmem', 0)
        if self.num_maskmem > 0:
            self.memory_feats = []
            self.directly_add_no_mem_embed = memory_cfg.get('directly_add_no_mem_embed', True)
            self.max_cond_frames_in_attn = memory_cfg.get('max_cond_frames_in_attn', -1)
            self.memory_temporal_stride_for_eval = memory_cfg.get('memory_temporal_stride_for_eval', 1)
            num_memory_attention_layer = memory_cfg.get('num_memory_attention_layer', 1)
            memory_attention_layer_cfg = memory_cfg.get('memory_attention_layer')
            memory_attention_layer_cls = globals().get(memory_attention_layer_cfg.pop('type'))
            self.memory_attention_layers = nn.ModuleList(
                [memory_attention_layer_cls(**memory_attention_layer_cfg) for _ in range(num_memory_attention_layer)]
            )
            self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, memory_cfg['no_mem_embed_dims'], 1, 1))
            trunc_normal_(self.no_mem_embed, std=0.02)

        self.memory_switch = True
    
    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame, current_feats_and_inputs, feat_sizes, output_dict, num_frames):
        # current_vision_feats: [[B,32,64,48], [B,64,32,24], [B,128,16,12], [B,256,8,6]]
        if self.num_maskmem == 0 or not self.memory_switch:  # Disable memory and skip fusion
            feats_and_inputs = current_feats_and_inputs
            return feats_and_inputs

        # Step 1: condition the visual features of the current frame on previous memories
        device = current_feats_and_inputs['keypoints_2d_cpn'].device
        if is_init_cond_frame:
            if self.directly_add_no_mem_embed:
                raise NotImplementedError
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                return pix_feat_with_mem
            else:
                feats_and_inputs = current_feats_and_inputs
                return feats_and_inputs
        else:
            to_cat_memory = []
            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
                )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    prev_frame_idx = frame_idx - t_rel
                else:
                    prev_frame_idx = ((frame_idx - 2) // stride) * stride
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases, so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)  # [B,1024,64,48]
                to_cat_memory.append(feats)

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.concatenate(to_cat_memory, dim=2)     # [B, T (a.k.a. num_maskmem), 71, 3]
        memory = memory.flatten(2).permute(0, 2, 1)     # [B, T*64*48, 1024]

        x = current_vision_feats[-1].flatten(2).permute(0, 2, 1)  # [B, 1024, 64, 48] -> [B, 1024, 64*48] -> [B, 64*48, 1024]
        for memory_attention_layer in self.memory_attention_layers:
            x = memory_attention_layer(
                x=x,            # [B, 64*48, 1024]
                memory=memory,  # [B, T*64*48, 1024]
                )
        C, H, W = feat_sizes[-1]
        pix_feat_with_mem = x.permute(0, 2, 1).reshape(-1, C, H, W)       # [B,3072,1024]->[B,1024,3072]->[B,1024,64,48]
        return pix_feat_with_mem    # [B,1024,64,48]

    def _track_step(self, frame_idx, is_init_cond_frame, current_feats_and_inputs, feat_sizes, output_dict, num_frames):
        current_out = {}
        feats_and_inputs = self._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_feats_and_inputs=current_feats_and_inputs,
            feat_sizes=feat_sizes,
            output_dict=output_dict,
            num_frames=num_frames,
            )
        # CAN BE WITH OR WITHOUT MEMORY
        # feats_and_inputs: {
        #       'vision_feats': [[B,32,64,48], [B,64,32,24], [B,128,16,12], [B,256,8,6]]
        #       'keypoints_2d_cpn': [B,17,2]
        #       'keypoints_2d_cpn_crop': [B,17,2]
        # }
        
        #TODO
        # ca_pf_outputs = self.volume_net(feats_and_inputs['keypoints_2d_cpn'], feats_and_inputs['keypoints_2d_cpn_crop'], feats_and_inputs['vision_feats']) # [B,1,17,3]
        keypoints_2d, ref, features_list = feats_and_inputs['keypoints_2d_cpn'], feats_and_inputs['keypoints_2d_cpn_crop'], feats_and_inputs['vision_feats']
        b, p, c = keypoints_2d.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.volume_net.coord_embed(keypoints_2d)  # x: [B,17,128]
        features_ref_list = [
            F.grid_sample(features, ref.unsqueeze(-2), align_corners=True).squeeze(-1).permute(0, 2, 1).contiguous() \
            for features in features_list]
        # Example: features=features_list[0]: [B,32,64,48] ==> grid_sample(ref.unsqueeze(-2):[B,17,1,2]) ==> [B,32,17,1] ==> [B,32,17] => [B,17,32]
        # [torch.Size([B, 17, 32]), torch.Size([B, 17, 64]), torch.Size([B, 17, 128]), torch.Size([B, 17, 256])]
        features_ref_list = [embed(features_ref_list[idx]) for idx, embed in enumerate(self.volume_net.feat_embed)]
        # [torch.Size([B, 17, 128]), torch.Size([B, 17, 128]), torch.Size([B, 17, 128]), torch.Size([B, 17, 128])]
        x = torch.stack([x,*features_ref_list], dim=1) # [b, 5, 17, 128]

        x += self.volume_net.Spatial_pos_embed
        x = self.volume_net.pos_drop(x)
        
        for blk in self.volume_net.context_blocks: # len=4. element blk: <class 'mvn.models.pose_dformer.DeformableBlock'>
            x = blk(x, ref, features_list)

        x = rearrange(x, 'b l p c -> (b p) l c')    # [B,5,17,128] => [B*17,5,128]
        for blk in self.volume_net.res_blocks: # 每个 blk 由 self-attention 和 MLP组成, 输入相当于 5 个 token, 每个 token 有 128 维特征, 然后做 self-attention, attention map 的形状为 5x5
            x = blk(x)

        x = rearrange(x, '(b p) l c -> b p (l c)', b=b)
        for blk in self.volume_net.joint_blocks:
            x = blk(x)

        ca_pf_outputs = self.volume_net.head(x).view(b, 1, p, -1)

        return current_out, ca_pf_outputs, feats_and_inputs

    def track_step(self, frame_idx, is_init_cond_frame, current_feats_and_inputs, feat_sizes, output_dict, num_frames, run_mem_encoder=True):
        current_out, ca_pf_outputs, feats_and_inputs = self._track_step(frame_idx, is_init_cond_frame, current_feats_and_inputs, feat_sizes, output_dict, num_frames)
        
        keypoints_3d_pred = ca_pf_outputs   # keypoints_3d_pred: [B,1,17,3]

        current_out['keypoints_3d_pred'] = keypoints_3d_pred

        self._encode_memory_in_output(current_feats_and_inputs, feat_sizes, run_mem_encoder, keypoints_3d_pred, current_out)
        
        return current_out
    
    def _encode_memory_in_output(self, current_feats_and_inputs, feat_sizes, run_mem_encoder, keypoints_3d_pred, current_out):
        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features = self._encode_new_memory(
                current_feats_and_inputs=current_feats_and_inputs,
                feat_sizes=feat_sizes,
                keypoints_3d_pred=keypoints_3d_pred,
            )
            current_out["maskmem_features"] = maskmem_features
        else:
            current_out["maskmem_features"] = None
    
    def _encode_new_memory(self, current_feats_and_inputs, feat_sizes, keypoints_3d_pred):
        maskmem_out = {"vision_features": keypoints_3d_pred}
        maskmem_features = maskmem_out["vision_features"]
        return maskmem_features
    
    def _prepare_backbone_features(self, backbone_out):
        # backbone_out = backbone_out.copy()
        feature_maps = backbone_out["backbone_hrnet"][-self.num_feature_levels:]    # [[B,T,32,64,48], [B,T,64,32,24], [B,T,128,16,12], [B,T,256,8,6]]
        feat_sizes = [x.shape[-3:] for x in feature_maps]
        vision_feats = feature_maps

        return backbone_out, vision_feats, feat_sizes
    
    def forward_tracking(self, backbone_out, videos, video_keypoints_2d_cpn, video_keypoints_2d_cpn_crop, return_dict=False):
        # videos: [B,T,256,192,3]
        # video_keypoints_2d_cpn: [B,T,17,2]
        # video_keypoints_2d_cpn_crop: [B,T,17,2]
        B, num_frames, H, W, _ = videos.shape
        _, vision_feats, feat_sizes = self._prepare_backbone_features(backbone_out)    # vision_feats: [[B,T,32,64,48], [B,T,64,32,24], [B,T,128,16,12], [B,T,256,8,6]]
        init_cond_frames = [0]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for frame_idx in range(num_frames):
            is_init_cond_frame = (frame_idx in init_cond_frames)

            current_feats_and_inputs = {
                'vision_feats': [x[:, frame_idx] for x in vision_feats],    # [[B,32,64,48], [B,64,32,24], [B,128,16,12], [B,256,8,6]]
                'keypoints_2d_cpn': video_keypoints_2d_cpn[:, frame_idx],   # [B,17,2]
                'keypoints_2d_cpn_crop': video_keypoints_2d_cpn_crop[:, frame_idx], # [B,17,2]
            }

            current_out = self.track_step(frame_idx, is_init_cond_frame, current_feats_and_inputs, feat_sizes, output_dict, num_frames)

            add_output_as_cond_frame = is_init_cond_frame
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][frame_idx] = current_out      # current_out: Dict. dict_keys(['keypoints_3d_pred', 'maskmem_features'])
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out

        if return_dict:
            return output_dict
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # # Make DDP happy with activation checkpointing by removing unused keys
        # all_frame_outputs = [
        #     {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        # ]
        return all_frame_outputs

    def forward_image(self, videos):
        B, T, H, W, _ = videos.shape
        videos_input = videos.reshape(B*T, H, W, 3).permute(0, 3, 1, 2).contiguous()    # [B,T,256,192,3] -> [B*T,3,256,192]
        features = self.backbone(videos_input)
        # [[BT,32,64,48], [BT,64,32,24], [BT,128,16,12], [BT,256,8,6]]
        features = [feat.reshape(B, T, *feat.shape[-3:]) for feat in features]
        # [[B,T,32,64,48], [B,T,64,32,24], [B,T,128,16,12], [B,T,256,8,6]]

        backbone_out = {
            'vision_features': features, 
            'backbone_hrnet': features,
        }
        return backbone_out

    def forward_sam2_style(self, videos, video_keypoints_2d_cpn, video_keypoints_2d_cpn_crop): # [B,T,256,192,3], [B,T,17,2], [B,T,17,2]
        device = video_keypoints_2d_cpn.device
        B, T, J, _ = video_keypoints_2d_cpn.shape
        video_keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        video_keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        backbone_out = self.forward_image(videos)
        previous_stages_out = self.forward_tracking(backbone_out, videos, video_keypoints_2d_cpn, video_keypoints_2d_cpn_crop)

        video_keypoints_3d_pred = []
        for frame_idx in range(T):
            keypoints_3d_pred = previous_stages_out[frame_idx]['keypoints_3d_pred']     # [B,1,17,3]
            video_keypoints_3d_pred.append(keypoints_3d_pred)
        video_keypoints_3d_pred = torch.cat(video_keypoints_3d_pred, dim=1)  # [B,T,17,3]

        return video_keypoints_3d_pred
    
    def forward_original(self, videos, video_keypoints_2d_cpn, video_keypoints_2d_cpn_crop): # [B,T,256,192,3], [B,T,17,2], [B,T,17,2]
        device = video_keypoints_2d_cpn.device
        B, T, J, _ = video_keypoints_2d_cpn.shape
        video_keypoints_2d_cpn_crop[..., :2] /= torch.tensor([192//2, 256//2], device=device)
        video_keypoints_2d_cpn_crop[..., :2] -= torch.tensor([1, 1], device=device)

        backbone_out = self.forward_image(videos)

        video_keypoints_3d_pred = self.volume_net(video_keypoints_2d_cpn, video_keypoints_2d_cpn_crop, backbone_out['vision_features'])

        return video_keypoints_3d_pred
    
    def forward(self, *args, **kwargs):
        return self.forward_sam2_style(*args, **kwargs)


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs