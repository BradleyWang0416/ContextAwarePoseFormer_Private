import torch
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn import TransformerDecoderLayer
import torch.nn.functional as F
from einops import rearrange

import torch.distributed as dist

from mvn.models import pose_hrnet
from mvn.models.networks import network
from mvn.models.cpn.test_config import cfg
from mvn.models.pose_dformer import PoseTransformer
# from mvn.utils.viz_skel_seq import viz_skel_seq_anim


class JointS1Loss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def smooth_l1_loss(self, pred, gt):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < self.beta
        loss = torch.where(cond, 0.5*l1_loss**2/self.beta, l1_loss-0.5*self.beta)
        return loss

    def forward(self, pred, gt):

        joint_dim = gt.shape[2] - 1
        visible = gt[..., joint_dim:]
        pred, gt = pred[..., :joint_dim], gt[..., :joint_dim]
 
        loss = self.smooth_l1_loss(pred, gt) * visible
        loss = loss.mean(dim=2).mean(dim=1).mean(dim=0)

        return loss


class MPJPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        # pred, gt: [B, K, 3] or [B, K, 4] where the last dim is visibility
        
        # 提取坐标和可见性
        joint_dim = gt.shape[-1] - 1 if gt.shape[-1] == 4 else gt.shape[-1]
        visible = gt[..., joint_dim:] if gt.shape[-1] == 4 else torch.ones_like(pred[..., :1])
        
        pred_coords = pred[..., :joint_dim]
        gt_coords = gt[..., :joint_dim]

        # 计算每个关节点的欧氏距离 (L2范数)
        # (pred - gt)**2 -> [B, K, 3]
        # torch.sum(..., dim=-1) -> [B, K]
        # torch.sqrt(...) -> [B, K]
        joint_errors = torch.sqrt(((pred_coords - gt_coords) ** 2).sum(dim=-1))

        # 应用可见性掩码
        loss = joint_errors * visible.squeeze(-1)

        # 对所有可见的关节点和样本取平均
        # 首先对关节点维度求和，然后除以可见关节点的总数
        loss = loss.sum() / visible.sum().clamp(min=1.0)

        return loss
    

class Tokenizer_loss(nn.Module):
    def __init__(self, joint_loss_w=1.0, e_loss_w=15.0, beta=0.05):
        super().__init__()

        # self.joint_loss = JointS1Loss(beta)
        self.joint_loss = MPJPELoss()
        self.joint_loss_w = joint_loss_w

        self.e_loss_w = e_loss_w

    def forward(self, output_joints, joints, e_latent_loss):

        losses = []
        joint_loss = self.joint_loss(output_joints, joints)
        joint_loss *= self.joint_loss_w
        losses.append(joint_loss)

        e_latent_loss *= self.e_loss_w
        losses.append(e_latent_loss)

        return losses


class MLPBlock(nn.Module):
    def __init__(self, dim, inter_dim, dropout_ratio):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ff(x)
    

class MixerLayer(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 hidden_inter_dim, 
                 token_dim, 
                 token_inter_dim, 
                 dropout_ratio):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.MLP_token = MLPBlock(token_dim, token_inter_dim, dropout_ratio)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.MLP_channel = MLPBlock(hidden_dim, hidden_inter_dim, dropout_ratio)

    def forward(self, x):
        y = self.layernorm1(x)
        y = y.transpose(2, 1)
        y = self.MLP_token(y)
        y = y.transpose(2, 1)
        z = self.layernorm2(x + y)
        z = self.MLP_channel(z)
        out = x + y + z
        return out
    

class PCT_Tokenizer(nn.Module):
    """ Tokenizer of Pose Compositional Tokens.
        paper ref: Zigang Geng et al. "Human Pose as
            Compositional Tokens"

    Args:
        stage_pct (str): Training stage (Tokenizer or Classifier).
        tokenizer (list): Config about the tokenizer.
        num_joints (int): Number of annotated joints in the dataset.
        guide_ratio (float): The ratio of image guidance.
        guide_channels (int): Feature Dim of the image guidance.
    """

    def __init__(self,
                 stage_pct,
                 tokenizer=dict(
                    guide_ratio=0.5,
                    ckpt="",
                    encoder=dict(
                        drop_rate=0.2,
                        num_blocks=4,
                        hidden_dim=512,
                        token_inter_dim=64,
                        hidden_inter_dim=512,
                        dropout=0.0,
                    ),
                    decoder=dict(
                        num_blocks=1,
                        hidden_dim=32,
                        token_inter_dim=64,
                        hidden_inter_dim=64,
                        dropout=0.0,
                    ),
                    codebook=dict(
                        token_num=34,
                        token_dim=512,
                        token_class_num=2048,
                        ema_decay=0.9,
                    ),
                    loss_keypoint=dict(
                        type='Tokenizer_loss',
                        joint_loss_w=1.0, 
                        e_loss_w=15.0,
                        beta=0.05,)
                 ),
                 num_joints=17,
                 guide_ratio=0,
                 guide_channels=0):
        super().__init__()

        self.stage_pct = stage_pct
        self.guide_ratio = guide_ratio
        self.num_joints = num_joints

        self.drop_rate = tokenizer['encoder']['drop_rate']
        self.enc_num_blocks = tokenizer['encoder']['num_blocks']
        self.enc_hidden_dim = tokenizer['encoder']['hidden_dim']
        self.enc_token_inter_dim = tokenizer['encoder']['token_inter_dim']
        self.enc_hidden_inter_dim = tokenizer['encoder']['hidden_inter_dim']
        self.enc_dropout = tokenizer['encoder']['dropout']

        self.dec_num_blocks = tokenizer['decoder']['num_blocks']
        self.dec_hidden_dim = tokenizer['decoder']['hidden_dim']
        self.dec_token_inter_dim = tokenizer['decoder']['token_inter_dim']
        self.dec_hidden_inter_dim = tokenizer['decoder']['hidden_inter_dim']
        self.dec_dropout = tokenizer['decoder']['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']
        self.token_dim = tokenizer['codebook']['token_dim']
        self.decay = tokenizer['codebook']['ema_decay']

        self.invisible_token = nn.Parameter(
            torch.zeros(1, 1, self.enc_hidden_dim))
        trunc_normal_(self.invisible_token, mean=0., std=0.02, a=-0.02, b=0.02)

        if self.guide_ratio > 0:
            self.start_img_embed = nn.Linear(
                guide_channels, int(self.enc_hidden_dim*self.guide_ratio))
        self.start_embed = nn.Linear(
            3, int(self.enc_hidden_dim*(1-self.guide_ratio)))
        
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim, 
                self.num_joints, self.enc_token_inter_dim,
                self.enc_dropout) for _ in range(self.enc_num_blocks)])
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)
        
        self.token_mlp = nn.Linear(
            self.num_joints, self.token_num)
        self.feature_embed = nn.Linear(
            self.enc_hidden_dim, self.token_dim)

        self.register_buffer('codebook', 
            torch.empty(self.token_class_num, self.token_dim))
        self.codebook.data.normal_()
        self.register_buffer('ema_cluster_size', 
            torch.zeros(self.token_class_num))
        self.register_buffer('ema_w', 
            torch.empty(self.token_class_num, self.token_dim))
        self.ema_w.data.normal_()        
        
        self.decoder_token_mlp = nn.Linear(
            self.token_num, self.num_joints)
        self.decoder_start = nn.Linear(
            self.token_dim, self.dec_hidden_dim)

        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 3)

        self.loss = Tokenizer_loss()

    def forward(self, joints, joints_feature, cls_logits, train=True):
        """Forward function. """

        if train or self.stage_pct == "tokenizer":
            # Encoder of Tokenizer, Get the PCT groundtruth class labels.
            joints_coord, joints_visible, bs = joints, torch.ones_like(joints[...,-1]).bool(), joints.shape[0]


            encode_feat = self.start_embed(joints_coord)
            if self.guide_ratio > 0:
                encode_img_feat = self.start_img_embed(joints_feature)
                encode_feat = torch.cat((encode_feat, encode_img_feat), dim=2)

            if train and self.stage_pct == "tokenizer":
                rand_mask_ind = torch.rand(
                    joints_visible.shape, device=joints.device) > self.drop_rate
                joints_visible = torch.logical_and(rand_mask_ind, joints_visible) 

            mask_tokens = self.invisible_token.expand(bs, joints.shape[1], -1)
            w = joints_visible.unsqueeze(-1).type_as(mask_tokens)
            encode_feat = encode_feat * w + mask_tokens * (1 - w)
                    
            for num_layer in self.encoder:  # num_layer: <class 'mvn.models.conpose_vqvae_byBrad.MixerLayer'>
                encode_feat = num_layer(encode_feat)
            encode_feat = self.encoder_layer_norm(encode_feat)
            
            encode_feat = encode_feat.transpose(2, 1)   # [B,17,512] -> [B,512,17]
            encode_feat = self.token_mlp(encode_feat).transpose(2, 1)   # Linear(in_features=17, out_features=34, bias=True)
            encode_feat = self.feature_embed(encode_feat).flatten(0,1)
            
            distances = torch.sum(encode_feat**2, dim=1, keepdim=True) \
                + torch.sum(self.codebook**2, dim=1) \
                - 2 * torch.matmul(encode_feat, self.codebook.t())
                
            encoding_indices = torch.argmin(distances, dim=1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self.token_class_num, device=joints.device)
            encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        else:
            bs = cls_logits.shape[0] // self.token_num
            encoding_indices = None
        
        if self.stage_pct == "classifier":
            part_token_feat = torch.matmul(cls_logits, self.codebook)
        else:
            part_token_feat = torch.matmul(encodings, self.codebook)

        if train and self.stage_pct == "tokenizer":
            # Updating Codebook using EMA
            dw = torch.matmul(encodings.t(), encode_feat.detach())

            """原始代码多GPU训练相关, 暂时用不到
            # sync
            n_encodings, n_dw = encodings.numel(), dw.numel()
            encodings_shape, dw_shape = encodings.shape, dw.shape
            combined = torch.cat((encodings.flatten(), dw.flatten()))
            dist.all_reduce(combined) # math sum
            sync_encodings, sync_dw = torch.split(combined, [n_encodings, n_dw])
            sync_encodings, sync_dw = \
                sync_encodings.view(encodings_shape), sync_dw.view(dw_shape)
            """

            # 直接使用本地计算的张量
            sync_encodings = encodings
            sync_dw = dw
            # --- 修改结束 ---

            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(sync_encodings, 0)
            
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.token_class_num * 1e-5) * n)
            
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * sync_dw
            self.codebook = self.ema_w / self.ema_cluster_size.unsqueeze(1)
            e_latent_loss = F.mse_loss(part_token_feat.detach(), encode_feat)
            part_token_feat = encode_feat + (part_token_feat - encode_feat).detach()
        else:
            e_latent_loss = None
        
        # Decoder of Tokenizer, Recover the joints.
        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)
        
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_joints = self.recover_embed(decode_feat)

        return recoverd_joints, encoding_indices, e_latent_loss

    def get_loss(self, output_joints, joints, e_latent_loss):
        """Calculate loss for training tokenizer.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output_joints (torch.Tensor[NxKx3]): Recovered joints.
            joints(torch.Tensor[NxKx3]): Target joints.
            e_latent_loss(torch.Tensor[1]): Loss for training codebook.
        """

        losses = dict()

        kpt_loss, e_latent_loss = self.loss(output_joints, joints, e_latent_loss)

        losses['joint_loss'] = kpt_loss
        losses['e_latent_loss'] = e_latent_loss

        return losses

    def init_weights(self, pretrained=""):
        """Initialize model weights."""

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            assert (self.stage_pct == "classifier"), \
                "Training tokenizer does not need to load model"
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)

            need_init_state_dict = {}

            for name, m in pretrained_state_dict['state_dict'].items():
                if 'keypoint_head.tokenizer.' in name:
                    name = name.replace('keypoint_head.tokenizer.', '')
                if name in parameters_names or name in buffers_names:
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=True)
        else:
            if self.stage_pct == "classifier":
                print('If you are training a classifier, '\
                    'must check that the well-trained tokenizer '\
                    'is located in the correct path.')
                
class CAPF_VQVAE(nn.Module):
    def __init__(self, config, device='cuda:0', processed_image_shape=(192,256)):
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

        # self.volume_net = PoseTransformer(config.model.poseformer, backbone=config.model.backbone.type)

        self.processed_image_shape = processed_image_shape
        self.tokenizer = PCT_Tokenizer(
            stage_pct='tokenizer', num_joints=17,
            guide_ratio=0.5, 
            guide_channels=32+64+128+256,   # pct_base: 1024; pct_huge: 2048
            )

    def forward(self, video_rgb, joint3d_image_affined, joint3d_image, img_ori_hw, train=True):
        # video_rgb: [B,T,256,192,3]
        # joint3d_image_affined: [B,T,17,3]
        # joint3d_image: [B,T,17,3]
        # img_ori_hw: [B,T,2]. element: (res_w, res_h)
        B, T, H, W, _ = video_rgb.shape

        video_rgb = video_rgb.permute(0, 1, 4, 2, 3).contiguous()  # [B,T,256,192,3] -> [B,T,3,256,192]

        device = joint3d_image_affined.device
        joint3d_image_affined_normed = joint3d_image_affined.clone()
        joint3d_image_affined_normed[..., :2] /= torch.tensor([self.processed_image_shape[0]//2, self.processed_image_shape[1]//2], device=device)
        joint3d_image_affined_normed[..., :2] -= torch.tensor([1, 1], device=device)
        joint3d_image_affined_normed[..., 2] = joint3d_image_affined_normed[..., 2] / img_ori_hw[..., 0:1] * 2

        joint3d_image_affined_normed_flattened = joint3d_image_affined_normed.reshape(-1, *joint3d_image_affined_normed.shape[2:])

        # TODO: joint3d_image_affined[..., 2:] 是否需要进行处理?

        with torch.no_grad():
            features_list = self.backbone(video_rgb.reshape(-1, *video_rgb.shape[2:]))
        # [[BT,32,64,48], [BT,64,32,24], [BT,128,16,12], [BT,256,8,6]]
        
        """从 2D grid_sample 到 3D grid_sample. 但是"每一帧都看到所有帧"好像不适用于 VQVAE 以及后续进一步用于 VLLM
        features_list = [feat.reshape(B, T, *feat.shape[-3:]) for feat in features_list]
        # [[B,T,32,64,48], [B,T,64,32,24], [B,T,128,16,12], [B,T,256,8,6]]

        joint_coords_xy = joint3d_image_affined[..., :2].clone()
        # 归一化时间坐标到 [-1, 1] 范围
        # 创建一个从 -1 到 1 的线性空间，代表时间维度
        time_coords = torch.linspace(-1.0, 1.0, T, device=device).view(1, T, 1, 1).expand(B, -1, self.num_joints, -1)

        # 将空间坐标和时间坐标合并
        # joint_coords_xy: [B, T, 17, 2]
        # time_coords: [B, T, 17, 1]
        grid_sample_ref_3d = torch.cat([time_coords, joint_coords_xy], dim=-1) # [B, T, 17, 3]. '3' means (t, y, x)

        features_ref_list = []
        for features in features_list:
            features_3d = features.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, Hf, Wf]
            grid = grid_sample_ref_3d.permute(0, 2, 1, 3) # [B, 17, T, 3]. '3' means (t, y, x)

            # 3. 执行3D Grid Sample
            # input: [B, C, T, Hf, Wf], grid: [B, 17, T, 3] -> output: [B, C, 17, T]
            features_ref = F.grid_sample(features_3d, grid, align_corners=True, padding_mode='border')
        """

        

        grid_sample_ref = joint3d_image_affined_normed_flattened[..., :2].unsqueeze(-2)  # [B*T,17,1,2]
        features_ref_list = []
        for features in features_list:
            features_ref = F.grid_sample(features, grid_sample_ref, align_corners=True)
            # features: [BT,256,8,6]; grid_sample_ref: [BT,17,1,2] >>> features_ref: [BT,256,17,1]
            # TODO: 如果 grid_sample_ref 的倒数第二维不是1, 而是3 (比如人体关节的特征不是一个点, 而是一个小区域), 会怎么样?
            features_ref = features_ref.squeeze(-1).permute(0, 2, 1).contiguous()   # [BT,17,256]
            features_ref_list.append(features_ref)
        # features_ref_list: [[BT,17,32], [BT,17,64], [BT,17,128], [BT,17,256]]

        joints_feature = torch.cat(features_ref_list, dim=-1)  # [BT,17,32+64+128+256]

        recoverd_joints, encoding_indices, e_latent_loss = self.tokenizer(
            joints=joint3d_image_affined_normed_flattened,
            joints_feature=joints_feature,
            cls_logits=None,
            train=train,
        )
        
        losses = self.tokenizer.get_loss(recoverd_joints, joint3d_image_affined_normed_flattened, e_latent_loss) if train else None
        # dict: {'joint_loss': ..., 'e_latent_loss': ...}
        
        return losses, recoverd_joints, joint3d_image_affined_normed_flattened
