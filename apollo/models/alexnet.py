from apollo import layers

def alexnet_layers():
    conv_weight_filler = layers.Filler(type="gaussian", std=0.01)
    bias_filler0 = layers.Filler(type="constant", value=0.0)
    bias_filler1 = layers.Filler(type="constant", value=1.0)
    conv_lr_mults = [1.0, 2.0]
    conv_decay_mults = [1.0, 0.0]

    alexnet_layers = [
        layers.Convolution(name="conv1", bottoms=["data"], param_lr_mults=conv_lr_mults,
            param_decay_mults=conv_decay_mults, kernel_size=11,
            stride=4, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=96),
        layers.ReLU(name="relu1", bottoms=["conv1"], tops=["conv1"]),
        layers.Pooling(name="pool1", bottoms=["conv1"], kernel_size=3, stride=2),
        layers.LRN(name="norm1", bottoms=["pool1"], local_size=5, alpha=0.0001, beta=0.75),
        layers.Convolution(name="conv2", bottoms=["norm1"], param_lr_mults=conv_lr_mults,
            param_decay_mults=conv_decay_mults, kernel_size=5,
            pad=2, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
        layers.ReLU(name="relu2", bottoms=["conv2"], tops=["conv2"]),
        layers.Pooling(name="pool2", bottoms=["conv2"], kernel_size=3, stride=2),
        layers.LRN(name="norm2", bottoms=["pool2"], local_size=5, alpha=0.0001, beta=0.75),
        layers.Convolution(name="conv3", bottoms=["norm2"], param_lr_mults=conv_lr_mults,
            param_decay_mults=conv_decay_mults, kernel_size=3,
            pad=1, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=384),
        layers.ReLU(name="relu3", bottoms=["conv3"], tops=["conv3"]),
        layers.Convolution(name="conv4", bottoms=["conv3"], param_lr_mults=conv_lr_mults,
            param_decay_mults=conv_decay_mults, kernel_size=3,
            pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=384),
        layers.ReLU(name="relu4", bottoms=["conv4"], tops=["conv4"]),
        layers.Convolution(name="conv5", bottoms=["conv4"], param_lr_mults=conv_lr_mults,
            param_decay_mults=conv_decay_mults, kernel_size=3,
            pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
        layers.ReLU(name="relu5", bottoms=["conv5"], tops=["conv5"]),
        layers.Pooling(name="pool5", bottoms=["conv5"], kernel_size=3, stride=2),
        layers.InnerProduct(name="fc6", bottoms=["pool5"], param_lr_mults=conv_lr_mults,
            param_decay_mults=conv_decay_mults,
            weight_filler=layers.Filler(type="gaussian", std=0.005),
            bias_filler=bias_filler1, num_output=4096),
        layers.ReLU(name="relu6", bottoms=["fc6"], tops=["fc6"]),
        layers.Dropout(name="drop6", bottoms=["fc6"], tops=["fc6"], dropout_ratio=0.5, phase='TRAIN'),
        layers.InnerProduct(name="fc7", bottoms=["fc6"], param_lr_mults=conv_lr_mults,
            param_decay_mults=conv_decay_mults,
            weight_filler=layers.Filler(type="gaussian", std=0.005),
            bias_filler=bias_filler1, num_output=4096),
        layers.ReLU(name="relu7", bottoms=["fc7"], tops=["fc7"]),
        layers.Dropout(name="drop7", bottoms=["fc7"], tops=["fc7"], dropout_ratio=0.5, phase='TRAIN'),
        layers.InnerProduct(name="fc8_flickr", bottoms=["fc7"], param_lr_mults=[10.0, 20.0],
            param_decay_mults=conv_decay_mults,
            weight_filler=layers.Filler(type="gaussian", std=0.01),
            bias_filler=bias_filler0, num_output=20),
        layers.SoftmaxWithLoss(name="loss", bottoms=["fc8_flickr", "label"]),
    ]

    return alexnet_layers
