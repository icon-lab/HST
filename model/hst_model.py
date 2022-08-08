## HST

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def divide_window(x, win_size=7):
    """
    Divide x into win_size by win_size windows.
    """
    batch, height, width, channel = x.shape
    height_num = height//win_size
    width_num = width//win_size
    x_reshaped = x.view(batch, height_num, win_size, width_num, win_size, channel).permute(0,1,3,2,4,5)
    divided_windows = x_reshaped.contiguous().view(-1, win_size, win_size, channel)
    return divided_windows


def undivide_window(divided_windows, height, width):
    """
    Reverse window division.
    """
    win_size = divided_windows.shape[1]
    height_num = height//win_size
    width_num = width//win_size
    batch = divided_windows.shape[0]//(height_num*width_num)
    x_reshaped = divided_windows.view(batch, height_num, width_num, win_size, win_size, -1).permute(0,1,3,2,4,5)
    x = x_reshaped.contiguous().view(batch, height_num*win_size, width_num*win_size, -1)
    return x

    
class PatchMerge(nn.Module):
    """
    Patch Merge Layer 
    """
    
    def __init__(self, in_res, dimension):
        super(PatchMerge, self).__init__()
        self.in_res = in_res
        self.dimension = dimension
        self.norm = nn.LayerNorm(4*dimension)
        self.reduce_tokens = nn.Linear(4*dimension, 2*dimension, bias=False)
        
    def forward(self, x):
        
        height, width = self.in_res
        W_B, num_patch, channel = x.shape
        x = x.view(W_B, height, width, channel)
        
        
        x_tl = x[:, 0::2, 0::2, :] # (top-left)
        x_bl = x[:, 1::2, 0::2, :] # (bottom-left)
        x_tr = x[:, 0::2, 1::2, :] # (top-right)
        x_br = x[:, 1::2, 1::2, :] # (bottom-right)
        
        x = torch.cat([x_tl, x_bl, x_tr, x_br], -1)  #(W_B, height/2, width/2, 4*channel)
        
        x = x.view(W_B, -1, 4*channel) #flatten height&width
        x = self.norm(x)
        x = self.reduce_tokens(x) # reduce the channel from 4*channel to 2*channel
        
        return x

       
class PatchPartition(nn.Module):
    """ 
    Partition the image into patch embeds
    """
    
    def __init__(self, img_size=224, h=4, img_channel=3, d=96):
        super(PatchPartition, self).__init__()      
        self.img_size = (img_size, img_size)
        self.patch_size = (h, h) 
        self.patch_res = (self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[1])
        self.num_patches = (self.img_size[1]//self.patch_size[1])*(self.img_size[0]//self.patch_size[0])
        
        self.img_channel = img_channel
        self.d = d
        
        self.projection  = nn.Conv2d(img_channel, d, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm_layer = nn.LayerNorm(d)
        
    def forward(self, x):

        batch_size, channel, height, width = x.shape
        
        
        x = self.projection(x) # (batch_size, 3, 224, 224) -> (batch_size, 96, 56, 56)
        x = x.flatten(2) # Patch Flatten (batch_size, 96, 56, 56) -> (batch_size, 96, 56*56)
        x = x.transpose(1, 2) # (batch_size, 96, 56*56) -> (batch_size, 56*56, 96)
        x = self.norm_layer(x)
        
        return x


class Mlp(nn.Module): 
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act_fn = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
        

class LWMSA(nn.Module):
    """
    Local windowed multi-head self-attention (LWMSA) module.
    """
    def __init__(self, win_size, dimension, num_attention_heads, use_bias=True, attn_dropout=0.0, proj_dropout=0.0):
        super(LWMSA, self).__init__()
        self.win_size = win_size 
        self.dimension = dimension
        self.num_attention_heads = num_attention_heads
        self.attn_head_size = int(dimension/num_attention_heads) 
        self.scale = (self.attn_head_size)**(-0.5)
        
        token_table_size = (2*win_size[0]-1, 2*win_size[1]-1)
        self.inter_token_interaction_matrix = nn.Parameter(torch.zeros(token_table_size[0]*token_table_size[1], num_attention_heads))
        
        self.qkv = nn.Linear(dimension, 3*dimension, bias=use_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dimension, dimension)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        vertical_coords = torch.arange(win_size[0])
        horizontal_coords = torch.arange(win_size[1])
        all_coords = torch.stack(torch.meshgrid([vertical_coords, horizontal_coords])) #(2,M,M)
        flattened_coords = torch.flatten(all_coords, 1)  #(2,M**2)
        relative_coords = flattened_coords[:,:,None] - flattened_coords[:,None,:] #(2,M**2,1)-(2,1,M**2)=(2,M**2,M**2)
        relative_coords = relative_coords.permute(1,2,0).contiguous() #(2,M**2,M**2) --> (M**2,M**2,2)
        relative_coords[:, :, 0] += win_size[0] - 1  #start from 0
        relative_coords[:, :, 1] += win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * win_size[1] - 1  #with respect to x-axis
        relative_position_index = relative_coords.sum(-1)  #(M**2,M**2)
        self.register_buffer("relative_position_index", relative_position_index)
        
        trunc_normal_(self.inter_token_interaction_matrix, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, attn_mask=None):
     
        W_B, num_patch, channel = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.view(W_B, num_patch, 3, self.num_attention_heads, int(channel/self.num_attention_heads))
        qkv = qkv.permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attention = torch.matmul(q,k.transpose(2,3))
      
        relative_position_bias_matrix = self.inter_token_interaction_matrix[self.relative_position_index.view(-1)].view(
            self.win_size[0]*self.win_size[1], self.win_size[0]*self.win_size[1],  -1)  
        relative_position_bias_matrix = relative_position_bias_matrix.permute(2, 0, 1).contiguous() 
        
        attention += relative_position_bias_matrix.unsqueeze(0)
        
        if attn_mask is not None: 
            batch = attn_mask.shape[0]
            attention = attention.view(int(W_B/batch), batch, self.num_attention_heads, num_patch, num_patch) + attn_mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_attention_heads, num_patch, num_patch)
            
        attention = self.softmax(attention)  
        attention = self.attn_dropout(attention) 
        
        x = torch.matmul(attention, v).transpose(1,2).reshape(W_B, num_patch, channel)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x 
        
        
class HST_Block(nn.Module):
    
    def __init__(self, dimension, in_res, num_attention_heads, win_size=7,
                 mlp_ratio=4.0, use_bias=True, dropout=0., attn_dropout=0., drop_path=0.,):
        super(HST_Block, self).__init__()  
        self.dimension = dimension
        self.win_size = win_size
        self.in_res = in_res
        self.num_attention_heads = num_attention_heads
        self.shift_siz1 = 0 # no shift for the 1st lwmsa module
        self.shift_siz2 = win_size//2 # shift for the 2nd lwmsa module
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate1 = drop_path[0] if isinstance(drop_path, list) else drop_path 
        self.drop_path_rate2 = drop_path[1] if isinstance(drop_path, list) else drop_path 

        if min(self.in_res) <= self.win_size:
            self.shift_siz1 = 0
            self.shift_siz2 = 0
            self.win_size = min(self.in_res)
        
        self.norm1 = nn.LayerNorm(dimension)
        self.attn1 = LWMSA(win_size=to_2tuple(self.win_size), dimension=dimension, 
            num_attention_heads=num_attention_heads, use_bias=use_bias, 
            attn_dropout=attn_dropout, proj_dropout=dropout)
        self.drop_path1 = DropPath(self.drop_path_rate1) if self.drop_path_rate1 > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dimension)
        mlp_dim = int(dimension*mlp_ratio)
        self.mlp1 = Mlp(in_features=dimension, hidden_features=mlp_dim, drop=dropout)

        if self.shift_siz2 > 0:
            # mask for cyclic-shift
            height, width = self.in_res
            mask_image = torch.zeros((1, height, width, 1))
            
            region = 0
            for h in (
                slice(0, -self.win_size),
                slice(-self.win_size, -self.shift_siz2),
                slice(-self.shift_siz2, None)):
                for w in (
                    slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_siz2),
                    slice(-self.shift_siz2, None)):
                    mask_image[:, h, w, :] = region
                    region += 1
                    
            mask_windows = divide_window(mask_image, self.win_size)
            mask_windows = mask_windows.view(-1, self.win_size*self.win_size)
            
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
                   
        self.register_buffer('attn_mask', attn_mask)

        self.norm3 = nn.LayerNorm(dimension)
        self.attn2 = LWMSA(win_size=to_2tuple(self.win_size), dimension=dimension, 
            num_attention_heads=num_attention_heads, use_bias=use_bias, 
            attn_dropout=attn_dropout, proj_dropout=dropout)
        self.drop_path2 = DropPath(self.drop_path_rate2) if self.drop_path_rate2 > 0.0 else nn.Identity()
        self.norm4 = nn.LayerNorm(dimension)
        self.mlp2 = Mlp(in_features=dimension, hidden_features=mlp_dim, drop=dropout)
        

    def forward(self, x):
        height, width = self.in_res
        batch, hw, channel = x.shape
        
        residual = x
        x = self.norm1(x)
        x = x.view(batch, height, width, channel) 
        new_x = x

        divided_x = divide_window(new_x, self.win_size)
        divided_x = divided_x.view(-1, self.win_size*self.win_size, channel) 
        
        local_attn_win1 = self.attn1(divided_x, attn_mask=None) 
        local_attn_win1 = local_attn_win1.view(-1, self.win_size, self.win_size, channel)
        
        new_x = undivide_window(local_attn_win1, height, width)
        x = new_x
        x = x.view(batch, height*width, channel)
        
        # Residual connections
        x = residual + self.drop_path1(x)
        x = x + self.drop_path1(self.mlp1(self.norm2(x)))
        
        # 2ND LWMSA MODULE
        residual2 = x
        x = self.norm3(x)
        x = x.view(batch, height, width, channel)
        
        # cyclic-shift 
        new_x = torch.roll(x, shifts=(-self.shift_siz2, -self.shift_siz2), dims=(1, 2))
        divided_x = divide_window(new_x, self.win_size) # (nW*B, M, M, C)
        divided_x = divided_x.view(-1, self.win_size*self.win_size, channel) # (nW*B, window_size*window_size, C)
        
        local_attn_win2 = self.attn2(divided_x, attn_mask=self.attn_mask) # (nW*B, window_size*window_size, C)
        local_attn_win2 = local_attn_win2.view(-1, self.win_size, self.win_size, channel)
        
        # reverse cylic_shift
        new_x = undivide_window(local_attn_win2, height, width)
        x = torch.roll(new_x, shifts=(self.shift_siz2, self.shift_siz2), dims=(1, 2))
        x = x.view(batch, height*width, channel)
        
        # Residual connections
        x = residual2 + self.drop_path2(x)
        x = x + self.drop_path2(self.mlp2(self.norm4(x)))

        return x


class HST_Layer(nn.Module):
    """
    HST Layer for one Stage.
    """
    
    def __init__(self, dimension, in_res, num_attention_heads, num_block, win_size=7, mlp_ratio=4.0, use_bias=True, 
                 dropout=0.0, attn_dropout=0.0, drop_path=0.0, patch_merge=None, use_checkpoint=False):
        super(HST_Layer, self).__init__()
        self.dimension = dimension
        self.in_res = in_res
        self.num_attention_heads = num_attention_heads
        self.num_block = num_block  
        self.use_checkpoint = use_checkpoint
        
        #HST Blocks
        self.HSTblocks = nn.Sequential(*[
            HST_Block(dimension=dimension, 
                                 in_res=in_res,
                                 num_attention_heads=num_attention_heads, 
                                 win_size=win_size,
                                 mlp_ratio = mlp_ratio,
                                 use_bias=use_bias,
                                 dropout=dropout, 
                                 attn_dropout=attn_dropout,
                                 drop_path=drop_path[2*i:2*i+2]) for i in range(num_block)])
        
        
        # Patch Merge
        if patch_merge is not None:
            self.patch_merge = patch_merge(in_res, dimension=dimension)
        else:
            self.patch_merge = None
            
    def forward(self, x):
        
        for HSTblock in self.HSTblocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(HSTblock, x)
            else:
                x = HSTblock(x)
        
        if self.patch_merge is not None:
            x = self.patch_merge(x)
        
        return x
   

class HSTModel(nn.Module):
    
    def __init__(self, img_size=224, h=4, img_channel=3, num_labels=2, d=96, num_blocks=[1, 1, 9, 1], 
                 num_attention_heads=[3, 6, 12, 24], win_size=7, mlp_ratio=4.0, use_bias=True,
                 dropout_rate=0.0, attn_dropout_rate=0.0, drop_path_rate=0.1, use_checkpoint=False, **kwargs):
        super(HSTModel, self).__init__()
        self.num_labels = num_labels
        self.S = len(num_blocks) # number of stages
        self.d = d  # embedding dimensionality
        self.feat_map_dim = int(d*2**(self.S-1)) 
        self.mlp_ratio = mlp_ratio

        # Partition image into non-overlapping patches
        self.patch_parts = PatchPartition(img_size=img_size, h=h, img_channel=img_channel, d=d)
        
        num_patches = self.patch_parts.num_patches
        patch_res = self.patch_parts.patch_res
        self.patch_res = patch_res
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        block_depths = [2*depth for depth in num_blocks]
        stoch_dd_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(block_depths))] # stochastic depth decay rule
        
        #build HST
        self.layers = nn.Sequential(*[
                HST_Layer(dimension=int(d*2**i), in_res=(patch_res[0]//(2**i), patch_res[1]//(2**i)), 
                num_attention_heads=num_attention_heads[i], num_block=num_blocks[i], win_size=win_size, 
                mlp_ratio=self.mlp_ratio, use_bias=use_bias, dropout=dropout_rate, attn_dropout=attn_dropout_rate, 
                drop_path=stoch_dd_rates[sum(block_depths[:i]):sum(block_depths[:i+1])], 
                patch_merge=PatchMerge if (i<self.S-1) else None, use_checkpoint=use_checkpoint)
                for i in range(self.S)
            ])
            
        self.norm_layer = nn.LayerNorm(self.feat_map_dim)
        self.avgpool_layer = nn.AdaptiveAvgPool1d(1)
        
        #Classification head
        self.head = nn.Linear(self.feat_map_dim, num_labels) if num_labels > 0 else nn.Identity()
        
        self.apply(self._init_weights)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    
    def forward(self, x):
        x = self.patch_parts(x)
        x = self.dropout(x)
        
        for hst_layer in self.layers:
            x = hst_layer(x)
            
        x = self.norm_layer(x)
        x = self.avgpool_layer(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
