from onpolicy.envs.mpe.market_graph_gym_env import MA_ELEC_Market_Graph_ENV

import numpy as np
import scipy.io as scio
import matlab.engine
def transtype(data):
    if type(data) is list:
        return matlab.double(data)
    else:
        return matlab.double(data.tolist())

if __name__ == '__main__':
    bus = scio.loadmat(r"D:\powerdist\on-policy-main\onpolicy\envs\mpe\case118zh\busdata.mat")["busdata"]


    #### 输入变量
    # 1、T: 当前所有边状态
    T = transtype(np.array([1]*117+[0]*15))
    # 2、PN: 节点有功负荷
    PN = transtype(bus[:,2]/10./2.)
    # 3、QN: 节点无功负荷
    QN = transtype(bus[:,3]/10./2.)

    [Q,R,L,V,X,Pg] = eng.Calc_Distflow(T, PN, QN, df, nargout = 6)

    print(np.array(Q).shape)