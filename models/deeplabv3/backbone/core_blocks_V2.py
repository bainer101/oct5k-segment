import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlockV1, self).__init__()
        
        # First Convolution (Same spatial size)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Second Convolution (Reduces spatial size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Identity (Skip Connection) - Adjust if needed
        self.shortcut = nn.Sequential()
        if stride == 2 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),  # Downsample & match channels
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)  # Adjust shortcut connection if needed

        # print(f"Identity block shape: {identity.shape}")

        x = self.conv1(x)
        # print(f"First conv layer output shape: {x.shape}")
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        # print(f"Second conv layer output shape: {x.shape}")
        x = self.batch_norm2(x)

        # print(f"Maxpool output shape {x.shape}")

        x += identity  # Add skip connection
        x = self.relu2(x)  # Apply activation after skip connection

        return x
    
class EncoderBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlockV2, self).__init__()

        # First Convolution (Same spatial size)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Second Convolution (Reduces spatial size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Identity (Skip Connection) - Adjust if needed
        self.shortcut = nn.Sequential()
        if stride == 2 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),  # Downsample & match channels
                nn.BatchNorm2d(out_channels)
            )
        self.shortcut_x1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False),  # Downsample & match channels
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # Identity path from input `x`

        # First convolution
        x1 = self.conv1(x)
        x1 = self.batch_norm1(x1)
        x1 = self.relu1(x1)  # Output of conv1 (to be used as skip connection)

        x1_skip = self.shortcut_x1(x1)

        # Second convolution
        x2 = self.conv2(x1)
        x2 = self.batch_norm2(x2)

        # Add identity connection from conv1 output (x1)
        x2 += identity + x1_skip  

        x2 = self.relu2(x2)  # Activation after skip connection

        return x2

class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class SkipConnectionBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipConnectionBlock1, self).__init__()
        
        # AvgPool2d with kernel_size=6 and stride=2 to reduce spatial dimensions
        self.average_pool = nn.AvgPool2d(kernel_size=4, stride=2)
        
        # 1x1 Convolution to increase the channel count
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0)
        
    def forward(self, x):
        x = self.average_pool(x)  # Reduces spatial dimensions
        x = self.conv1x1(x)        # Adjusts the channels to match out_channels (e.g., 32)
        return x

class SkipConnectionBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipConnectionBlock2, self).__init__()
        
        # AvgPool2d with kernel_size=6 and stride=2 to reduce spatial dimensions
        self.average_pool = nn.AvgPool2d(kernel_size=3, stride=2)
        
        # 1x1 Convolution to increase the channel count
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0)
        
    def forward(self, x):
        x = self.average_pool(x)  # Reduces spatial dimensions
        x = self.conv1x1(x)        # Adjusts the channels to match out_channels (e.g., 32)
        return x

class InterpolateLayer(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=True):
        super(InterpolateLayer, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Block 1
        self.interpolate = InterpolateLayer(size=(45, 28), mode='bilinear', align_corners=True)
        self.deconv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=0)
        
        # Block 2
        self.interpolate1 = InterpolateLayer(size=(93, 59), mode='bicubic', align_corners=True)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0)
        
        # Block 3
        self.interpolate2 = InterpolateLayer(size=(189, 121), mode='bicubic', align_corners=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=0)
        
        # Block 4
        self.interpolate3 = InterpolateLayer(size=(382, 246), mode='bicubic', align_corners=True)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # Block 5
        self.interpolate4 = InterpolateLayer(size=(752, 480), mode='bicubic', align_corners=True)
        self.deconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=0)
        
        # Block 6
        self.deconv5 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0)
        self.deconv6 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=5, stride=1, padding=0)
        self.deconv7 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        # Assuming x has shape (batch_size, 256, height, width)
        
        # Block 1
        x = F.interpolate(x, size=(45, 28), mode='bilinear', align_corners=True)
        x = self.deconv(x)

        # Block 2
        x = F.interpolate(x, size=(93, 59), mode='bicubic', align_corners=True)
        x = self.deconv1(x)

        # Block 3
        x = F.interpolate(x, size=(189, 121), mode='bicubic', align_corners=True)
        x = self.deconv2(x)

        # Block 4
        x = F.interpolate(x, size=(382, 246), mode='bicubic', align_corners=True)
        x = self.deconv3(x)

        # Block 5
        x = F.interpolate(x, size=(752, 480), mode='bicubic', align_corners=True)
        x = self.deconv4(x)

        # Block 6
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, output_dim=1024):
        super(TransformerBlock, self).__init__()
        
        self.input_dim = input_dim  # Expected input dimensions [batch_size, 256, 45, 36]
        self.token_dim = (9, 9)  # Each token has dimensions 9x9
        self.channels = input_dim[0]  # Number of channels, 256 in this case
        
        # Check if height and width are divisible by the token size
        if input_dim[1] % self.token_dim[0] != 0 or input_dim[2] % self.token_dim[1] != 0:
            raise ValueError("Input dimensions (height, width) must be divisible by token dimensions")

        # Calculate the number of tokens based on channels, height, and width
        self.num_tokens = self.channels * (input_dim[1] // self.token_dim[0]) * (input_dim[2] // self.token_dim[1])
        
        # Positional embedding for each token
        self.positional_embedding = nn.Parameter(torch.randn(self.num_tokens, embed_dim))
        
        # Linear layer to project each token to the embedding dimension
        self.linear = nn.Linear(self.token_dim[0] * self.token_dim[1], embed_dim)
        
        # Multihead Attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

        # Layer Normalization for stability
        self.norm = nn.LayerNorm(embed_dim)

        # Final linear layer to project the output to the desired dimension (output_dim)
        self.output_projection = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # Input shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.shape
        print(x.shape)

        # Reshape input into patches (tokens), each patch of size 9x9 in the spatial dimensions
        x = x.unfold(2, self.token_dim[0], self.token_dim[0])  # Unfold height
        x = x.unfold(3, self.token_dim[1], self.token_dim[1])  # Unfold width

        # Now x has shape [batch_size, channels, num_patches_h, num_patches_w, token_dim[0], token_dim[1]]
        num_patches_h = x.shape[2]
        num_patches_w = x.shape[3]
        x = x.contiguous().view(batch_size, self.num_tokens, self.token_dim[0] * self.token_dim[1])  # [batch_size, 5120, 81]
        
        # Linear layer to embed each token
        x = self.linear(x)  # [batch_size, 5120, embed_dim]
        x_original = x.clone()

        # Add positional embedding (broadcasting over batch dimension)
        x = x + self.positional_embedding.unsqueeze(0)  # [batch_size, 5120, embed_dim]
        
        # Prepare for Multihead Attention
        x = x.permute(1, 0, 2)  # [5120, batch_size, embed_dim]
        
        # Apply Multihead Attention
        x, _ = self.multihead_attention(x, x, x)  # [5120, batch_size, embed_dim]
        
        # Fix skip connection shape issue
        x = self.norm(x + x_original.permute(1, 0, 2))  

        # Pool across tokens using max pooling for a single representation per batch item
        x, _ = x.max(dim=0)  # [batch_size, embed_dim]

        # Project to the desired output dimension
        x = self.output_projection(x)  # [batch_size, output_dim]

        return x  # Output shape: [batch_size, output_dim]
    

class SPPModule(nn.Module):
    def __init__(self, pool_mode='avg', sizes=(1, 2, 3, 6)):
        super().__init__()
        if pool_mode == 'avg':
            pool_layer = nn.AdaptiveAvgPool2d
        elif pool_mode == 'max':
            pool_layer = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError

        self.pool_blocks = nn.ModuleList([
            nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x

# Add this above TransformerBlockV2 class
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        self.sigmoid_channel = nn.Sigmoid()

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        scale = self.sigmoid_channel(avg_out + max_out).view(b, c, 1, 1)
        x = x * scale

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg, max_], dim=1)
        scale = self.spatial(spatial)
        x = x * scale
        return x

    
class TransformerBlockV2(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, output_dim=1024, reduced_channels=64):
        super(TransformerBlockV2, self).__init__()

        self.token_dim = (9, 9)  # Patch size

        pool_size = (1, 2, 3)  # Pool sizes for SPP

        # Spatial Pyramid Pooling for feature aggregation
        self.avg_spp = SPPModule('avg', pool_size)
        self.max_spp = SPPModule('max', pool_size)

        _channels = input_dim[0] * int(sum([x ** 2 for x in pool_size]))

        # Reduce channels after SPP (aggregated features → reduced_channels)
        self.channel_reduction = nn.Conv2d(_channels, reduced_channels, kernel_size=1, padding=0, bias=False)

        # self.channel_reduction = nn.Sequential(
        #     nn.Conv2d(_channels, _channels // 2, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(_channels // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(_channels // 2, reduced_channels, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(reduced_channels),
        #     nn.ReLU(inplace=True),
        # )

        # Compute number of tokens (reduced_channels instead of 256)
        self.num_tokens = reduced_channels * (input_dim[1] // self.token_dim[0]) * (input_dim[2] // self.token_dim[1])
        
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(self.num_tokens, embed_dim))

        # Linear layer to embed patches
        self.linear = nn.Linear(self.token_dim[0] * self.token_dim[1], embed_dim)

        # Multihead Attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

        # Normalization
        self.norm = nn.LayerNorm(embed_dim)

        # self.cbam = CBAM(256)

        # Output projection
        self.output_projection = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        batch_size, channels, height, width = x.shape  # [BS, 256, 45, 36]
        # print(f"Heigth {height}")
        # print(f"Width {width}")

        assert channels == 256, "Input channels must be 256"
        # assert height == 45 and width == 36, "Input dims do not match"

        # x = self.cbam(x)

        # Apply Spatial Pyramid Pooling
        x = self.avg_spp(x) + self.max_spp(x)

        # print(f'SPP output shape: {x.shape}')

        # Reduce channels using 1×1 Conv1D (Preserving spatial hierarchy)
        x = self.channel_reduction(x)  # [BS, reduced_channels, spatial_feature_dim]

        # print(f'Channel reduction shape: {x.shape}')

        # We want to expand the tensor to match the required spatial dimensions for unfolding
        x = x.expand(batch_size, x.shape[1], width, height)  # Expanding to [BS, reduced_channels, 45, 36]

        # print(f'Expanded shape: {x.shape}')
        
        # Unfold the feature map into patches
        x = x.unfold(2, self.token_dim[0], self.token_dim[0])  # [BS, reduced_channels, num_patches_h, _, 9]
        x = x.unfold(3, self.token_dim[1], self.token_dim[1])  # [BS, reduced_channels, num_patches_h, num_patches_w, 9, 9]

        num_patches_h, num_patches_w = x.shape[2], x.shape[3]
        # Flatten the patches into tokens
        x = x.contiguous().view(batch_size, self.num_tokens, self.token_dim[0] * self.token_dim[1])  # [BS, num_tokens, 81]

        # print(f'Patches shape: {x.shape}')

        # Linear embedding
        x = self.linear(x)  # [BS, num_tokens, embed_dim]

        # **Add Positional Embeddings**
        x = x + self.positional_embedding.unsqueeze(0)  # [BS, num_tokens, embed_dim]

        # Store the original x before attention for skip connection
        x_original = x.clone()

        # Prepare for Multihead Attention
        x = x.permute(1, 0, 2)  # [num_tokens, BS, embed_dim]

        # Apply Multihead Self-Attention
        attn_output, _ = self.multihead_attention(x, x, x)

        # **Skip Connection & LayerNorm**
        x = self.norm(attn_output + x_original.permute(1, 0, 2))

        x_pooled, _ = x.max(dim=0)  # Pool over tokens
        x_pooled = self.output_projection(x_pooled)  # [BS, output_dim]

        return x_pooled

        