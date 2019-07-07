
import gen_deploy
import gen_solver
# gen train and val prototxt file
gen_deploy. write_net()
gen_solver.write_solver()

import caffe
caffe.set_mode_cpu()
solver = caffe.SGDSolver('/home/lx/PycharmProjects/demo_mnist/models/solver.prototxt')
solver.solve()

