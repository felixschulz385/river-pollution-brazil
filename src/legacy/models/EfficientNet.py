import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Dropout, concatenate, Input,\
    Add, Activation, UpSampling2D, Multiply, GlobalAveragePooling2D, Reshape, Dense, DepthwiseConv2D, Concatenate, SeparableConv2D, ReLU

## 3: EfficientNet
def EfficientNet(n_classes=1, im_sz=416, n_channels=3, drop_connect_rate=0.2, phi = 5):
    
    def swish(x):
        """
        Applies the Swish activation function element-wise to a tensor.
        The Swish activation function is a non-linear activation function that has been proposed as an
            alternative to traditional activation functions like ReLU. 
        It is defined as Swish(x) = x * sigmoid(x), where sigmoid is the sigmoid activation function.

        Args:
            x (tensor): Input tensor to apply the Swish activation function.

        Returns:
            tensor: Output tensor with the Swish activation applied.
        """
        return x * tf.math.sigmoid(x)

    def SEBlock(input_tensor, ratio=4):
        """
        Creates a Squeeze-and-Excitation (SE) block.
        The Squeeze-and-Excitation (SE) block is a mechanism to adaptively recalibrate channel-wise feature
            responses, emphasizing important channels and suppressing less important ones

        Args:
            input_tensor (tensor): Input tensor to the SE block.
            reduction_ratio (int): Reduction ratio for the SE block. It determines the number of filters
                in the intermediate fully connected layers.
        """
        channels = input_tensor.shape[-1]
        se_shape = (1, 1, channels)
        
        se = GlobalAveragePooling2D()(input_tensor)
        se = Reshape(se_shape)(se)
        se = Dense(channels // ratio, activation=swish)(se)
        se = Dense(channels, activation='sigmoid')(se)
        
        return Multiply()([input_tensor, se])
    
    def ConvBNBlock(x, filters, kernel_size, stride):
        """
        This function applies a series of convolution, batch normalization, 
        and a custom swish activation function to the input tensor.
        
        Parameters:
        - x (tf.Tensor): The input tensor to the block. 
                        Shape should be (batch_size, height, width, channels).
        
        - filters (int): The number of filters for the convolutional layer.
        
        - kernel_size (tuple): The size of the convolutional kernel. 
                            A tuple (height, width).
        
        - stride (int): The stride for the convolutional layer.
        
        Returns:
        - tf.Tensor: The output tensor after applying convolution, 
                    batch normalization, and swish activation. 
                    Shape will be (batch_size, new_height, new_width, filters).
        """
        x = Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        return swish(x)

    def MBConvBlock(input_tensor, in_channels, out_channels, expand_ratio, stride, se_ratio, drop_connect_rate):
        """
        Creates a Mobile Inverted Bottleneck Convolution (MBConv) block.
        MBConv is a fundamental building block in EfficientNet models, combining depthwise separable convolution
            with expansion and optionally SE mechanism.

        Args:
            input_tensor (tensor): Input tensor to the MBConv block.
            out_filters (int): Number of output filters (channels) of the block.
            kernel_size (int): Size of the convolutional kernel (e.g., 3 for a 3x3 kernel).
            expansion_factor (int): Expansion factor for the depthwise convolution.
                It determines the number of filters in the expansion phase.
            stride (int): Stride for the depthwise convolution (typically 1 or 2).
            use_se (bool): Whether to include the Squeeze-and-Excitation (SE) block in the MBConv.
                If True, SE block is added to enhance feature representation.
            id_skip (bool): Whether to include a skip connection (identity mapping) from input to output.
                If True, a skip connection is added to facilitate gradient flow.

        Returns:
            tensor: Output tensor of the MBConv block.
            
        References:
        - M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks." [https://arxiv.org/abs/1801.04381]
        """
        x = input_tensor
        
        # Expansion phase
        if expand_ratio != 1:
            x = ConvBNBlock(x, in_channels * expand_ratio, (1, 1), stride=1)
        
        # Depthwise convolution phase
        x = DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = swish(x)
        
        # Squeeze and excitation phase
        if se_ratio:
            x = SEBlock(x, se_ratio)
        
        # Output phase
        x = Conv2D(out_channels, (1, 1), strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        
        if drop_connect_rate:
            x = Dropout(drop_connect_rate)(x)
        
        return x
    
    def MBDeConvBlock(input_tensor, skip, filters, ):
        """
        This function performs upsampling, concatenation with a skip connection, and
        further convolution and batch normalization on the input tensor for the
        purpose of semantic segmentation tasks.
        
        Parameters:
        - input_tensor (tf.Tensor): The input tensor from the previous layer in the 
                                    decoder, or the bottleneck layer if it's the 
                                    first block. 
                                    Shape should be (batch_size, height, width, channels).
        
        - skip (tf.Tensor): The skip connection tensor from the corresponding encoder 
                            block.
                            Shape should be (batch_size, skip_height, skip_width, skip_channels).
                            
        - filters (int): The number of filters for the convolutional layers in the block.
        
        Returns:
        - tf.Tensor: The output tensor after performing upsampling, concatenation with
                    skip connection, and two rounds of convolution and batch normalization.
                    Shape will be (batch_size, new_height, new_width, filters).
        """
        # Make sure the number of filters matches for concatenation
        #skip = Conv2D(filters, (1, 1), padding='same', activation=None)(skip)
        
        # Up-sampling
        x = UpSampling2D((skip.shape[1] / input_tensor.shape[1], skip.shape[2] / input_tensor.shape[2]))(input_tensor)
        x = Conv2D(filters, (1, 1), padding='same', activation=None)(x)
        
        # Concatenate with skip connection
        x = Concatenate()([x, skip])
        x = BatchNormalization()(x)
        
        # Additional Conv layers to refine the features
        x = ConvBNBlock(x, filters, (3, 3), stride=1)
        x = ConvBNBlock(x, filters, (3, 3), stride=1)
        
        if drop_connect_rate:
            x = Dropout(drop_connect_rate)(x)
        
        return x
    
    def scale_efficientnet_config(base_config, phi):
        """
        Scale the configuration of an EfficientNet architecture based on a given phi value.
        
        Parameters:
            base_config (list of dict): The configuration for the base EfficientNet-B0 model.
                Each dictionary in the list should contain:
                - 'in_channels' (int): The number of input channels for the block.
                - 'out_channels' (int): The number of output channels for the block.
                - 'expand_ratio' (int): The expansion ratio for the block.
                - 'stride' (int): The stride for the block.
                - 'se_ratio' (float): The squeeze-and-excitation ratio for the block.
                - 'num_repeat' (int): The number of times to repeat the block.
                
            phi (int): The phi value for scaling the width, depth, and resolution according to the EfficientNet paper.
                A higher value will result in a larger and more powerful network, but it will also be slower to train.
                
        Returns:
            scaled_config (list of dict): The scaled configuration for the EfficientNet architecture.
        """
        
        alpha = 1.2  # width multiplier
        beta = 1.1   # depth multiplier
        gamma = 1.15 # resolution multiplier (not used in this example)

        # Calculate scaled values
        alpha = alpha ** phi
        beta = beta ** phi
        gamma = gamma ** phi

        scaled_config = []

        for config in base_config:
            scaled_config.append({
                'in_channels': int(config['in_channels'] * alpha),
                'out_channels': int(config['out_channels'] * alpha),
                'expand_ratio': config['expand_ratio'],
                'stride': config['stride'],
                'se_ratio': config['se_ratio'] # Optionally add the number of times to repeat the block
            })

        return scaled_config
    
    # Configuration for MBConv blocks
    base_config = [
        {'in_channels': 32, 'out_channels': 16, 'expand_ratio': 1, 'stride': 1, 'se_ratio': 0.25},
        {'in_channels': 16, 'out_channels': 24, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
        {'in_channels': 24, 'out_channels': 40, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
        {'in_channels': 40, 'out_channels': 80, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
        {'in_channels': 80, 'out_channels': 112, 'expand_ratio': 6, 'stride': 1, 'se_ratio': 0.25},
        {'in_channels': 112, 'out_channels': 192, 'expand_ratio': 6, 'stride': 2, 'se_ratio': 0.25},
        {'in_channels': 192, 'out_channels': 320, 'expand_ratio': 6, 'stride': 1, 'se_ratio': 0.25}
    ]
    
    config_list = scale_efficientnet_config(base_config, phi)
    
    # Initial convolution
    input_img = Input(shape=(im_sz, im_sz, n_channels))
    x = ConvBNBlock(input_img, config_list[0]["in_channels"], (3, 3), stride=2)
    
    # Encoder
    skip_connections = []
    for config in config_list:
        x = MBConvBlock(x, **config, drop_connect_rate=drop_connect_rate)
        if config['stride'] == 1:
            skip_connections.append(x)
    
    # Decoder
    for i, filters in enumerate(reversed(list([x["in_channels"] for x in config_list if x["stride"] == 1]))):
        x = MBDeConvBlock(x, skip_connections[-(i+1)], filters)

    # Output
    x = UpSampling2D((2, 2))(x)
    output = Conv2D(n_classes, (1, 1), activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=input_img, outputs=output)
    
    return model

if __name__ == "__main__":
    model = EfficientNet()
    model.summary()