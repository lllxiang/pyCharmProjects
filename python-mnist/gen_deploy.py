
from caffe import layers as L, params as P, to_proto

dataPath = '/home/lx/PycharmProjects/demo_mnist/data/'
modelPath = '/home/lx/PycharmProjects/demo_mnist/models/'
train_lmdb = dataPath + 'mnist_train_lmdb'
val_lmdb = dataPath + 'mnist_test_lmdb'

mean_file = modelPath + 'mean.binaryproto'
train_proto = modelPath + 'train.prototxt'
val_proto = modelPath + 'val.prototxt'


def create_net(lmdb, batch_size, include_acc=False):

    data, label = L.Data(source=lmdb, name='mnist', backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                         transform_param=dict(scale=0.00390625))

    conv1 = L.Convolution(data, kernel_size=5, stride=1, num_output=20, pad=0, weight_filler=dict(type='xavier'))
    pool1 = L.Pooling(conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)


    conv2 = L.Convolution(pool1, kernel_size=5, stride=1, num_output=50, pad=0, weight_filler=dict(type='xavier'))
    pool2 = L.Pooling(conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    fc3 = L.InnerProduct(pool2, num_output=500, weight_filler=dict(type='xavier'))
    relu3 = L.ReLU(fc3, in_place=True)

    fc4 = L.InnerProduct(relu3, num_output=10, weight_filler=dict(type='xavier'))


    loss = L.SoftmaxWithLoss(fc4, label)

    if include_acc:
        acc = L.Accuracy(fc4, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)


def write_net():
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_lmdb, batch_size=64)))

    with open(val_proto, 'w') as f:
        f.write(str(create_net(val_lmdb, batch_size=32, include_acc=True)))
    print 'gen deploy done!'


if __name__ == '__main__':

    write_net()