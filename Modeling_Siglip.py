import torch
import torch.nn as nn
from typing import Optional, Tuple

"""
    Number of patch in an image of 224 pixels:
        # of patches = (224/16)x(224/16) = 196
    So, the image becomes a sequence of 196 patch embeddings, which
    the transformer can process like a sequence of tokens.
"""
    
class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768, # embedding dimension
        intermediate_size= 4 * 768, # Used in FFN/MLP
        num_hidden_layers=12, # 12 transformer encoder layers (also called blocks).
        num_attention_heads=12,
        num_channels=3, # RGB
        image_size=224, # 224 x 224
        patch_size=16, # Image is split into non-overlapping patches of size 16x16 pixels
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int=None, # num_image_tokens indicates how many image patch embeddings (tokens) the model will have/create for each input image.
        **kwargs
    ):

        super().__init__()

        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.num_channels=num_channels
        self.image_size=image_size
        self.patch_size=patch_size
        self.layer_norm_eps=layer_norm_eps
        self.attention_dropout=attention_dropout
        self.num_image_tokens=num_image_tokens


class SiglipVisionEmbeddings(nn.Module): # PatchEmbedding
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
            f"Image dimensions ({height}x{width}) must be divisible by patch size ({self.patch_size})"

        self.config=config # Not mandatory for this code
        self.embed_dim=config.hidden_size
        self.num_channels=config.num_channels
        self.image_size=config.image_size
        self.patch_size=config.patch_size
        self.patch_embedding=nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )
        self.num_patches=(self.image_size // self.patch_size)**2,
        self.num_positions=self.num_patches
        self.position_embedding=nn.Embedding(self.num_positions, self.embed_dim) # here positional encoding/embedding learned compared to fixed (sinusoidal) in vanilla transformer.
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)), # Shape: [1, Num_patches]
            persistent=False,
        )

        # In the vanilla Transformer, sinusoidal positional encodings are used to inject sequence order into the model, which is inherently permutation-invariant. These encodings 
        # are fixed and based on sinusoidal functions of different frequencies. They enable the model to generalize to sequences longer than those seen during training, thanks to 
        # their mathematical regularity and extrapolation capabilities. This makes them theoretically elegant, especially for tasks involving variable-length input.

        # In contrast, Vision Transformers (ViTs) operate on image patches arranged in a fixed 2D grid structure. Since image inputs are of fixed or bounded size, ViTs can afford 
        # to use learned positional embeddings. These are trainable vectors associated with each patch location, which are often simpler to implement and have been found empirically 
        # effective for vision tasks.
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor: # The pixel_values tensor represents the raw input image(s), not patches.
        _, _, height, width=pixel_values.shape # pixel_values.shape: [batch_size, channels, height, width]
        
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size.
        # The output of the convolution will have shape [Batch_size, Embed_dim, Num_Patches_H, Num_Patches_W]
        # Where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        patch_embeds=self.patch_embedding(pixel_values) # Patches created here

        # Flatten the spatial dimensions (Num_Patches_H, Num_Patches_W) 
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # Where Num_Patches = Num_Patches_H * Num_Patches_W; e.g., = (224/16) * (224/16) = 14 * 14 = 196
        # Flattening a zero-dimensional tensor will return a one-dimensional view.
        embeddings = patch_embeds.flatten(start_dim=2) # Keep dimensions 0 and 1 (Batch_Size, Embed_Dim) unchanged and flatten dimensions 2 and onward (Num_Patches_H, Num_Patches_W) into one. 

        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)

        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim].
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # Output shape: [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipAttention(nn.Module): 
    # Multi-head attention from 'Attention Is All You Need' paper.
    def __init__(self, config):
        super().__init__()
        self.config=config # Not mandatory for this code
        self.embed_dim=config.hidden_size
        self.num_heads=config.num_attention_heads
        self.head_dim=self.embed_dim // self.num_heads
        self.scale=self.head_dim**-0.5 # Equivalent to 1/sqrt(self.head_dim)
        self.dropout=config.attention_dropout
        self.q_proj=nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj=nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj=nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj=nn.Linear(self.embed_dim, self.embed_dim) # output head

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Shape of hidden_states: [Batch_Size, Num_Patches (Context_Length or Sequence_Length), Embed_Dim] 
        batch_size, seq_len, _ = hidden_states.shape

        # Shape of query [Batch_Size, Num_Patches (Context_Length or Sequence_Length), Embed_Dim]
        query=self.q_proj(hidden_states)

        # Shape of key [Batch_Size, Num_Patches (Context_Length or Sequence_Length), Embed_Dim]
        key=self.k_proj(hidden_states)

        # Shape of value [Batch_Size, Num_Patches (Context_Length or Sequence_Length), Embed_Dim]
        value=self.v_proj(hidden_states)

        # Shape of query/key/value [Batch_Size, Num_Heads, Num_Patches (Context_Length or Sequence_Length), Head_Dim]
        query=query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key=key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value=value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate the attention score using the formula Q * K^T / sqrt(d_k).
        # Shape of attention score [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attention_score=(query @ key.transpose(2, 3)) * self.scale

        if attention_score.shape != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but this is {attention_score.shape}"
            )
    
        # Apply the softmax row-wise.
        # Shape of attention_weights [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attention_weights=nn.functional.softmax(attention_score, dim=-1, dtype=torch.float32).to(query.dtype)

        # Apply dropout only during training
        attention_weights=nn.functional.dropout(attention_weights, p=self.dropout, training=self.training)

        # Multiply the attention_weights by the value to get context_vector
        # Shape of context_vector [Batch_Size, Num_Heads, Num_patches, Head_Dim]
        context_vector=attention_weights @ value

        if context_vector.shape != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Context_vector should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is {context_vector.shape}"
            )
        
        #[Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        context_vector=context_vector.transpose(1, 2).contiguous()

        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        context_vector=context_vector.reshape(batch_size, seq_len, self.embed_dim) # Reshape concatenates output from each head

        #[Batch_Size, Num_Patches, Embed_Dim]
        context_vector=self.out_proj(context_vector)

        return context_vector, attention_weights


class SiglipMLP(nn.Module): # FeedForwardNetwork (FFN)
    def __init__(self, config):
        super().__init__()
        self.config=config # Not mandatory for this code
        self.fc1=nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2=nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states=self.fc1(hidden_states) # Expansion
        
        # [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states=nn.functional.gelu(hidden_states, approximate="tanh") # There is no rule of thumb to choose nonlinearity just heuristic

        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states=self.fc2(hidden_states) # Contraction

        return hidden_states
    


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim=config.hidden_size
        self.self_attn=SiglipAttention(config)
        self.layer_norm1=nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp=SiglipMLP(config)
        self.layer_norm2=nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Shape of residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual=hidden_states

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states=self.layer_norm1(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states += residual

        # Shape of residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual=hidden_states

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states=self.layer_norm2(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states=self.mlp(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states += residual

        return hidden_states



class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config=config # Not mandatory for this code
        self.layers=nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Shape of input_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states=inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states=encoder_layer(hidden_states)

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config=config # Not mandatory for this code
        self.embed_dim=config.hidden_size
        self.embeddings=SiglipVisionEmbeddings(config)
        self.encoder=SiglipEncoder(config)
        self.post_layernorm=nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states=self.embeddings(pixel_values)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        last_hidden_state=self.encoder(inputs_embeds=hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        last_hidden_state=self.post_layernorm(last_hidden_state)

        return last_hidden_state
    

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.vision_model=SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
