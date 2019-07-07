from caffe.proto import caffe_pb2


def write_solver():
    s = caffe_pb2.SolverParameter()
    modelPath = '/home/lx/PycharmProjects/demo_mnist/models/'
    solver_file=modelPath+'solver.prototxt'
    s.train_net = modelPath+'train.prototxt'
    s.test_net.append(modelPath+'val.prototxt')


    s.test_interval = 938
    s.test_iter.append(100)
    s.max_iter = 9380
    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.lr_policy = 'step'
    s.stepsize=3000
    s.gamma = 0.1
    s.display = 20
    s.snapshot = 938
    s.snapshot_prefix =modelPath+'lenet'
    s.type ='SGD'
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    with open(solver_file, 'w') as f:
        f.write(str(s))