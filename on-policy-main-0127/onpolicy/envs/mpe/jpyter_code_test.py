import numpy as np
import scipy.io as scio
import matlab.engine
# from onpolicy.envs.mpe.market_graph_gym_env import *
eng = matlab.engine.start_matlab()



def transtype(data):
    if type(data) is list:
        return matlab.double(data)
    else:
        return matlab.double(data.tolist())

if __name__ == '__main__':
    #
    df = eng.environment()

    bus = scio.loadmat("./case118zh/busdata.mat")["busdata"]

    # 测试
    ptr_vec_1 = np.load('ptr_vec1.npy')

    t1 = np.load("T1.npy")

    ptr_vec_1 = ptr_vec_1.reshape(-1, 1)

    [ptr_matrix1, FTR1, g1] = eng.Calc_P2P(transtype(t1), transtype(ptr_vec_1), df,  nargout = 3)

    print(np.array(ptr_matrix1))

    T = transtype(np.array([1] * 117 + [0] * 15))
    # 2、PN: 节点有功负荷
    PN = transtype(bus[:, 2] / 10. / 2.)
    print('PN', len(bus[:, 2]))
    # 3、QN: 节点无功负荷
    print('QN',len(bus[:, 3]))
    QN = transtype(bus[:, 3] / 10. / 2.)

    [Q, R, L, V, X, Pg] = eng.Calc_Distflow(T, PN, QN, df, nargout=6)