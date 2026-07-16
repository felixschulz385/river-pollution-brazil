import tensorflow as tf
from model_utils import EinsumLayer

## 4: EfficientTransformer
def EfficientTransformer(input_shape = (416, 416, 3), num_classes = 1, patch_size = 8, 
                         d_model = 32, depth = 1, num_heads = 2, window_size = 26): 
    """
    Build the EfficientTransformer for semantic segmentation.
    Model structure largely follows EfficientTransformer from reference below. Adaptions to 
      cater to hardware restrictions have been made. Edge enhancement is not implemented.

    Parameters:
    - input_shape: tuple, shape of the input image.
    - num_classes: int, number of output classes.
    - patch_size: int, size of the patches.
    - d_model: int, depth of the model.
    - depth: int, number of transformer layers.
    - num_heads: int, number of attention heads.
    - window_size: int, size of the window for local attention.
    
    References:
    - Xu, Z. et al., "Efficient Transformer for Remote Sensing Image Segmentation." [https://doi.org/10.3390/rs13183585]
    """
    
    def patch_embed(x, dim, patch_size, downsample):
        """
        Perform patch embedding on input tensor x.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).
            dim (int): Dimension of the output embedding.
            patch_size (int): Size of the patch to be extracted.
            downsample (int): Stride for patch extraction.

        Returns:
            tf.Tensor: Patch embedded tensor.
        """
        x = tf.keras.layers.Conv2D(dim, kernel_size=patch_size, strides=downsample, padding='same')(x)
        return x

    def position_embed(x, dim, dim_mult):
        """
        Perform position embedding on input tensor x.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).
            dim (int): Dimension of the output embedding.

        Returns:
            tf.Tensor: Position embedded tensor.
        """
        dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.Conv2D(dim * dim_mult, 1, padding='same')(dwconv)
        return x
    
    def window_attention(x, dim, window_size, num_heads):
        """
        Perform window-based self-attention on input tensor x.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).
            dim (int): Dimension of the output embedding.
            window_size (int): Size of the attention window.
            num_heads (int): Number of attention heads.

        Returns:
            tf.Tensor: Self-attention output tensor.
        """
        
        def rearrange_windows(x, window_shape):
            B, H, W, C = x.shape
            # Reshape the tensor into a grid of windows
            x = tf.reshape(x, [-1, H // window_shape[0], window_shape[0], W // window_shape[1], window_shape[1], C])
            # Transpose the grid to prepare for attention
            x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
            # Flatten the grid into patches
            return tf.reshape(x, [-1, (H // window_shape[0] * W // window_shape[1]), window_shape[0] * window_shape[1], C])
        
        x_shape = x.shape
        B, L, C, = -1, x_shape[1], x_shape[2]
        H = W = int(x_shape[1] ** 0.5)
        num_windows = (H // window_size) * (W // window_size)
        
        # Rearrange the input tensor into non-overlapping patches
        x = tf.reshape(x, [B, H, W, C])
        x = rearrange_windows(x, [window_size, window_size])

        # Project the input patches to queries, keys, and values
        scale = (dim // num_heads) ** -0.5
        qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)(x)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q *= scale
        
        # Reshape queries, keys, and values for multi-head attention
        q = tf.reshape(q, [B, num_windows, window_size ** 2, num_heads, C // num_heads])
        k = tf.reshape(k, [B, num_windows, window_size ** 2, num_heads, C // num_heads])
        v = tf.reshape(v, [B, num_windows, window_size ** 2, num_heads, C // num_heads])
        
        # compute attention for this window
        attn = EinsumLayer('bwmhd,bwnhd->bwhnm')((q, k))
        attn = tf.nn.softmax(attn, axis=-1)
        
        # perform attention-weighted aggregation for this window
        out = EinsumLayer('bwhmm,bwnhd->bwhnd')((attn, v))
        out = tf.transpose(out, [0, 1, 3, 2, 4])
        out = tf.reshape(out, [B, num_windows, window_size * window_size, C])
        
        """
        ### Legacy implementation of regular (non-windowed) transformer
        
        # Reshape queries, keys, and values for multi-head attention
        q = tf.reshape(q, [B, U, num_heads, C // num_heads])
        k = tf.reshape(k, [B, U, num_heads, C // num_heads])
        v = tf.reshape(v, [B, U, num_heads, C // num_heads])

        # Transpose for efficient batched matrix multiplication
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Compute attention scores using matrix multiplication
        attn = EinsumLayer('bhnd,bhmd->bhnm')((q, k))
        attn = tf.nn.softmax(attn, axis=-1)
        attn = tf.keras.layers.Dropout(0.1)(attn)

        # Perform attention-weighted aggregation
        out = EinsumLayer('bhnm,bhmd->bhnd')((attn, v))
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, U, C])
        """

        # Reshape the output to its original format
        out = tf.reshape(out, [B, H // window_size, W // window_size, window_size, window_size, C])
        out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
        out = tf.reshape(out, [B, H, W, C])
        out = tf.reshape(out, [B, L, C])
        
        # Apply a dense layer to the output
        out = tf.keras.layers.Dense(dim)(out)
        out = tf.keras.layers.Dropout(0.1)(out)
        return out    
    
    def mlp(x, in_dim, hidden_dim):
        """
        Multi-Layer Perceptron (MLP) block.

        Args:
            x (tf.Tensor): Input tensor.
            in_dim (int): Input dimension of the MLP.
            hidden_dim (int): Hidden dimension of the MLP.

        Returns:
            tf.Tensor: MLP output tensor.
        """
        x = tf.keras.layers.Dense(hidden_dim)(x)
        x = tf.keras.layers.Activation('gelu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(in_dim)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        return x

    def transformer_block(input, dim, depth, num_heads, window_size):
        """
        Transformer block. Implements one transformer layer in given depth with
        residual connection and mlp activation.
        
        Args:
            input (tf.Tensor): Input tensor.
            dim (int): Dimension of the transformer block.
            depth (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            window_size (int): Size of the attention window.

        Returns:
            tf.Tensor: Transformer block output tensor.
        """
        # Flatten
        x = tf.reshape(input, (-1, input.shape[1] * input.shape[2], dim))
        for _ in range(depth):
            skip = x
            x = skip + window_attention(x, dim, window_size, num_heads)
            x = x + mlp(x, dim, dim)
        # Restore spatial resolution
        x = tf.reshape(x, [-1, input.shape[1], input.shape[2], dim])
        return x 
    
    ###
    # Stage definitions
    ###
    
    def stem(x, dim=d_model):
        """
        Stem block for the EfficientTransformer model.
        Implements initial convolution to reduce complexity

        Args:
            x (tf.Tensor): Input tensor.
            dim (int): Dimension of the stem block.

        Returns:
            tf.Tensor: Stem block output tensor.
        """
        x = tf.keras.layers.Conv2D(dim, (7, 7), strides=2, padding='same', activation='relu')(inputs)
        return x
    
    def transformer_stage(x, dim=d_model, downsample=1, dim_mult=1, depth=depth, num_heads=num_heads, 
                            window_size=window_size, patch_size=patch_size):
        """
        Transformer stage in the EfficientTransformer model.
        Implements patch embedding, position embedding and transformer block.

        Args:
            x (tf.Tensor): Input tensor.
            dim (int): Dimension of the transformer stage.
            downsample (int): Downsampling factor.
            depth (int): Number of transformer blocks in the stage.
            num_heads (int): Number of attention heads.
            window_size (int): Size of the attention window.
            patch_size (int): Size of the patch.

        Returns:
            tf.Tensor: Transformer stage output tensor.
        """
        # Patch Embedding
        x = patch_embed(x, dim, patch_size, downsample)
        # Position Embedding
        x = position_embed(x, dim, dim_mult)
        # Transformer Block (Embedding + Encoder)
        x = transformer_block(x, dim * dim_mult, depth, num_heads, window_size)
        return x
    
    def mlphead_stage(input, dim=d_model, add=None, out_shape=input_shape):
        """
        MLP head stage in the EfficientTransformer model.
        Implements MLP head stage with optional skip connection.

        Args:
            input (tf.Tensor): Input tensor.
            dim (int): Dimension of the MLP head stage.
            add (tf.Tensor or None): Tensor to be added to the output.
            out_shape (tuple): Desired output shape.

        Returns:
            tf.Tensor: MLP head stage output tensor.
        """
        # Flatten
        e = tf.reshape(input, (-1, input.shape[1] * input.shape[2], input.shape[3]))
        # MLP
        e = mlp(e, e.shape[-1], e.shape[-1])
        # Reshape
        e = tf.reshape(e, (-1, input.shape[1], input.shape[2], input.shape[3]))
        # Add
        if add is not None:
            # Add
            e += add
            # MLP
            e = mlp(e, e.shape[-1], e.shape[-1])
        # Reduce Dimensions
        e = tf.keras.layers.Conv2D(filters=int(e.shape[-1]/2), kernel_size=1, strides=1)(e) #tf.cast(tf.cast(e.shape[-1], "float") / 2, "int32")
        # Upsample for output
        d = tf.keras.layers.Conv2D(filters=dim, kernel_size=1, strides=1)(e)
        d = tf.keras.layers.UpSampling2D((out_shape[0] / d.shape[1], out_shape[1] / d.shape[2]))(d)
        # Upsample for skip
        e = tf.keras.layers.UpSampling2D()(e)

        return d, e

    ###
    # Model
    ###
        
    inputs = tf.keras.Input(shape=input_shape)
    
    # Stem
    x = stem(inputs)
    c1 = transformer_stage(x)
    c2 = transformer_stage(c1, downsample=2, dim_mult = 2)
    c3 = transformer_stage(c2, downsample=2, dim_mult = 4)
    c4 = transformer_stage(c3, downsample=2, dim_mult = 8)
        
    # Segmentation Head
    d4, e4 = mlphead_stage(c4)
    d3, e3 = mlphead_stage(c3, add = e4)
    d2, e2 = mlphead_stage(c2, add = e3)
    d1, _ = mlphead_stage(c1, add = e2)
    
    # Output
    output = d4 + d3 + d2 + d1
    output = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=3, strides=1, padding='same', activation='sigmoid')(output)
    
    return tf.keras.Model(inputs=inputs, outputs=output)