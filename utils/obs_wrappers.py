# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import gym
from gym.spaces.box import Box
import omegaconf
import json
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle
from torchvision.utils import save_image
import hydra
from einops import repeat
from r3meval.utils.vit import vit_s16
from r3meval.utils.vdual import VDual
from r3meval.utils.transformer import RMSNorm, SwishGLU
from r3meval.utils.video_transformer import SpaceTimeTransformer 
from r3meval.utils.parse_config import ConfigParser
from r3meval.utils.util import state_dict_data_parallel_fix,state_dict_data_parallel_fix1,inflate_positional_embeds,state_dict_data_parallel_fix2,state_dict_data_parallel_fix3
from transformers import AutoModel
import torch.nn.functional as F
def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("first_decoder_layer" in key) or ("decoder_layer" in key) or ("decoder" in key) or ("vid_proj" in key) or ("vid_proj1" in key) or ("vid_proj2" in key) or ("vid_proj3" in key) or ("resnet_proj" in key) or ("resnet_proj1" in key) or ("module.norm" in key) or ("encoder_layer" in key) or ("encoder" in key):
        #if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def _get_embedding(embedding_name='resnet34', load_path="", *args, **kwargs):
    if load_path == "random":
        prt = False
    else:
        prt = True
    if embedding_name == 'resnet34':
        model = models.resnet34(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet18':
        model = models.resnet18(pretrained=prt, progress=False)
        embedding_dim = 512
    elif embedding_name == 'resnet50':
        model = models.resnet50(pretrained=prt, progress=False)
        embedding_dim = 2048
    else:
        print("Requested model not available currently")
        raise NotImplementedError
    # make FC layers to be identity
    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    model.fc = Identity()
    model = model.eval()
    return model, embedding_dim


class ClipEnc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, im):
        e = self.m.encode_image(im)
        return e


class StateEmbedding(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.

    From https://pytorch.org/vision/stable/models.html
    All pre-trained models expect input images normalized in the same way,
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    where H and W are expected to be at least 224.

    Args:
        env (Gym environment): the original environment,
        embedding_name (str, 'baseline'): the name of the convolution model,
        device (str, 'cuda'): where to allocate the model.

    """
    def __init__(self, env, embedding_name=None, device='cuda', load_path="", proprio=0, camera_name=None, env_name=None):
        gym.ObservationWrapper.__init__(self, env)

        self.proprio = proprio
        self.load_path = load_path
        self.start_finetune = False
        if load_path == "clip":
            import clip
            model, cliptransforms = clip.load("RN50", device="cuda")
            embedding = ClipEnc(model)
            embedding.eval()
            embedding_dim = 1024
            self.transforms = cliptransforms
        elif (load_path == "random") or (load_path == ""):
                embedding, embedding_dim = _get_embedding(embedding_name=embedding_name, load_path=load_path)
                self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif "r3m" == load_path:
            from r3m import load_r3m_reproduce
            rep = load_r3m_reproduce("r3m")
            rep.eval()
            embedding_dim = rep.module.outdim
            embedding = rep
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()]) # ToTensor() divides by 255
        elif "ours" == load_path:
            model = models.resnet50(pretrained=False, progress=False)
            model.fc=Identity()
            if torch.cuda.is_available():
                   device = "cuda"
            else:
                   device = "cpu"
            embedding_dim = 2048
            # model = torch.nn.DataParallel(model)
            checkpoint =  torch.load('/home/mscsim/Yixiao/r3m88-2.pth', map_location=torch.device(device))#for original model /home/mscsim/.r3m/original_r3m/model.pt
            #checkpoint = checkpoint['r3m']#for original model
            checkpoint=remove_language_head(checkpoint)
            checkpoint=state_dict_data_parallel_fix2(checkpoint, model.state_dict()) #for our model
            #checkpoint=state_dict_data_parallel_fix3(checkpoint, model.state_dict())  #for original model
            # new_state_dict = {}
            # for key, value in checkpoint.items():
            #   if key.startswith('module.convnet'):
            #        new_key = key.replace('module.convnet', 'module')
            #        new_state_dict[new_key] = value
            #   else:
            #        new_state_dict[key] = value
            model.load_state_dict(checkpoint,strict=True)
            model.eval()   
            embedding=model 
            self.transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif load_path=="mvp":
            self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            #model,embedding_dim=vit_s16('/home/mscsim/Yixiao/vit_small_baseline.pth',map_location='cuda:{}'.format(0)) #baseline
            model,embedding_dim=vit_s16('/home/mscsim/Yixiao/mvp88.pth',map_location='cuda:{}'.format(0))
            # checkpoint =  torch.load('/home/mscsim/Yixiao/mvp88.pth', map_location='cuda:{}'.format(0))
            # checkpoint=remove_language_head(checkpoint)
            # checkpoint=state_dict_data_parallel_fix2(checkpoint, model.state_dict())
            # model.load_state_dict(checkpoint,strict=True)
            model.eval()   
            embedding=model 
        elif load_path=="egovlp":
            self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            embedding_dim=768
            model = SpaceTimeTransformer(num_frames=1,
                                        time_init='zeros',
                                        attention_style='frozen-in-time',
                                        num_classes=0)
            model.head = nn.Identity()
            model.pre_logits = nn.Identity()    
            #model.eval() 
            # self.text_model = AutoModel.from_pretrained('distilbert-base-uncased',
            #        cache_dir='/home/mscsim/Yixiao/distilbert-base-uncased')
            # self.video_model = model
        
            # self.video_model.fc = nn.Identity()
            checkpoint = torch.load('/home/mscsim/Yixiao/egovlp84.pth', map_location='cuda:{}'.format(0))
            state_dict = checkpoint
            new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
            new_state_dict = state_dict_data_parallel_fix1(new_state_dict, model.state_dict())
            new_state_dict = inflate_positional_embeds(model.state_dict(),new_state_dict)
            model.load_state_dict(new_state_dict, strict=False) 
            model.eval()   
            embedding=model  
        elif load_path=="voltron":
            with open('/home/mscsim/Yixiao/cache/v-dual/v-dual-config.json', "r") as f:
                model_kwargs = json.load(f)
                if "hf_cache" in model_kwargs:
                   model_kwargs["hf_cache"] = '/home/mscsim/Yixiao/cache/hf-cache'
            model=VDual(**model_kwargs)
            embedding_dim=384
            checkpoint,_=torch.load('/home/mscsim/Yixiao/cache/v-dual/v-dual.pt',map_location='cuda:{}'.format(0))
            model.load_state_dict(checkpoint, strict=True)
            model.eval()   
            embedding=model   
            self.transforms = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), # ToTensor() divides by 255
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            raise NameError("Invalid Model")
        embedding.eval()

        if device == 'cuda' and torch.cuda.is_available():
            print('Using CUDA.')
            device = torch.device('cuda')
        else:
            print('Not using CUDA.')
            device = torch.device('cpu')
        self.device = device
        embedding.to(device=device)

        self.embedding, self.embedding_dim = embedding, embedding_dim
        self.observation_space = Box(
                    low=-np.inf, high=np.inf, shape=(self.embedding_dim+self.proprio,))

    def observation(self, observation):
        ### INPUT SHOULD BE [0,255]
        if self.embedding is not None:
            inp = self.transforms(Image.fromarray(observation.astype(np.uint8))).reshape(-1, 3, 224, 224)
            if "r3m" in self.load_path:
                ## R3M Expects input to be 0-255, preprocess makes 0-1
                inp *= 255.0
            if "egovlp" in self.load_path:
                inp=inp.unsqueeze(1)
            if "voltron" in self.load_path:
                inp=torch.cat([inp,inp],dim=0)
                inp=torch.unsqueeze(inp,0)
            inp = inp.to(self.device)
            if "voltron" in self.load_path:
                with torch.no_grad():
                      emb=self.embedding(inp,mode="visual")
                emb=torch.mean(emb,dim=1)
                emb=emb.view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        
            else: 
                with torch.no_grad():
                    emb = self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()

            ## IF proprioception add it to end of embedding
            if self.proprio:
                try:
                    proprio = self.env.unwrapped.get_obs()[:self.proprio]
                except:
                    proprio = self.env.unwrapped._get_obs()[:self.proprio]
                emb = np.concatenate([emb, proprio])

            return emb
        else:
            return observation

    def encode_batch(self, obs,obs_first, finetune=False):
        ### INPUT SHOULD BE [0,255]
        inp = []
        # obs_first = self.transforms(Image.fromarray(obs_first.astype(np.uint8))).reshape(-1, 3, 224, 224)
        for o,of in zip(obs,obs_first):
            i = self.transforms(Image.fromarray(o.astype(np.uint8))).reshape(-1, 3, 224, 224)
            of = self.transforms(Image.fromarray(of.astype(np.uint8))).reshape(-1, 3, 224, 224)
            if "r3m" in self.load_path:
                ## R3M Expects input to be 0-255, preprocess makes 0-1
                i *= 255.0
            if "egovlp" in self.load_path:
                i=i.unsqueeze(1)
            if "voltron" in self.load_path:
                i=torch.cat([of,i],dim=0)
                i=torch.unsqueeze(i,0)
            inp.append(i)
        inp = torch.cat(inp)
        inp = inp.to(self.device)
        if finetune and self.start_finetune:
            emb = self.embedding(inp).view(-1, self.embedding_dim)
        else:
           if "voltron" in self.load_path:
                with torch.no_grad():
                   emb=self.embedding(inp,mode="visual")
                Mappooling= MAPBlock(n_latents=1, embed_dim=384, n_heads=6)
                Mappooling=Mappooling.to('cuda')
                emb=emb.to('cuda')
                emb=Mappooling(emb)
                # emb=emb.view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
           else: 
                with torch.no_grad():
                  emb =  self.embedding(inp).view(-1, self.embedding_dim).to('cpu').numpy().squeeze()
        return emb

    def get_obs(self):
        if self.embedding is not None:
            return self.observation(self.env.observation(None))
        else:
            # returns the state based observations
            return self.env.unwrapped.get_obs()
          
    def start_finetuning(self):
        self.start_finetune = True
     
    def _inflate_positional_embeds(curr_state_dict, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(curr_state_dict.keys())
        if 'temporal_embed' in new_state_dict and 'temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = 1
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded model has MORE frames than current...'
                          f'### loading weights, filling in the extras via')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via')
                    # if self.load_temporal_fix == 'zeros':
                    #     new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                    #     new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    # elif self.load_temporal_fix in ['interp', 'bilinear']:
                    #     # interpolate
                    #     # unsqueeze so pytorch thinks its an image
                    #     mode = 'nearest'
                    #     if self.load_temporal_fix == 'bilinear':
                    #         mode = 'bilinear'
                    load_temporal_embed = load_temporal_embed.unsqueeze(0)
                    new_temporal_embed = F.interpolate(load_temporal_embed,
                                                     (curr_num_frames, embed_dim), mode='bilinear', align_corners=True).squeeze(0)
                     
                new_state_dict['temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'pos_embed' in new_state_dict and 'pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = curr_state_dict['pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict
class MAPAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        """Multi-Input Multi-Headed Attention Operation"""
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Projections (no bias) --> separate for Q (seed vector), and KV ("pool" inputs)
        self.q, self.kv = nn.Linear(embed_dim, embed_dim, bias=False), nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, seed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        (B_s, K, C_s), (B_x, N, C_x) = seed.shape, x.shape
        assert C_s == C_x, "Seed vectors and pool inputs must have the same embedding dimensionality!"

        # Project Seed Vectors to `queries`
        q = self.q(seed).reshape(B_s, K, self.n_heads, C_s // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B_x, N, 2, self.n_heads, C_x // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention --> compute weighted sum over values!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        attn = scores.softmax(dim=-1)
        vals = (attn @ v).transpose(1, 2).reshape(B_s, K, C_s)

        # Project back to `embed_dim`
        return self.proj(vals)
class MAPBlock(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, 2 * n_heads

        # Projection Operator
        self.projection = nn.Linear(embed_dim, self.embed_dim)

        # Initialize Latents
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.embed_dim))
        nn.init.normal_(self.latents, std=0.02)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = MAPAttention(self.embed_dim, n_heads=self.n_heads)

        # Position-wise Feed-Forward Components
        self.mlp_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)), nn.GELU())
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = repeat(self.latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])
        latents = self.attn_norm(latents + self.attn(latents, self.projection(x)))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)

class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        if "v2" in env.spec.id:
            self.get_obs = env._get_obs

    def get_image(self):
        if self.camera_name == "default":
            print("Camera not supported")
            assert(False)
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                            device_id=self.device_id)
        else:
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                              camera_name=self.camera_name, device_id=self.device_id)
        img = img[::-1,:,:]
        return img

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()
        