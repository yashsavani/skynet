def get_hyper():
    param = {}
    # The train/test net protocol buffer definition
    param['net_prototxt'] = "examples/mnist/lenet_train_test.prototxt"
    # test_iter specifies how many forward passes the test should carry out.
    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
    # covering the full 10,000 testing images.
    param['test_iter'] = 100
    # Carry out testing every 500 training iterations.
    param['test_interval'] = 500
    # The base learning rate, momentum and the weight decay of the network.
    param['base_lr'] = 0.01
    param['momentum'] = 0.9
    param['weight_decay'] = 0.0005
    param['gamma'] = 1.0
    param['stepsize'] = 10000
    # Display every 100 iterations
    param['display_interval'] = 100
    # The maximum number of iterations
    param['max_iter'] = 10000
    # snapshot intermediate results
    param['snapshot_interval'] = 5000
    param['snapshot_prefix'] = "/tmp"
    return param
