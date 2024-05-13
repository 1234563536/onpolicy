import pandas as pd
import numpy as np
from gurobipy import *

def PyDistflow(T, PN, QN, params):

    # 读取常量
    busNum = params["busNum"]
    baseMVA = params["baseMVA"] # 功率基准值为10MVA
    basekV = params["basekV"]   # 电压基准值为11kV
    baseI = params["baseI"]     # 电流基准值
    I_max = params["I_max"]     # 电流最大值
    V_max = params["V_max"]     # 节点电压上限
    V_min  = params["V_min"]    # 节点电压下限
    c = params["c"]             # 发电成本

    allbranch = params["allbranch"]
    bus_num = params["bus_num"]
    gen_num = params["gen_num"]
    genP_max = params["genP_max"]
    genQ_max =params["genQ_max"]
    commen_num =  params["commen_num"]

    branch = allbranch[np.nonzero(T)]

    ## 请检查数值转换逻辑是否正确
    P_i = {i+1:PN[i] for i in bus_num-1} # 节点有功功率
    Q_i = {i+1:QN[i] for i in bus_num-1} # 节点无功功率

    f = branch[:, 0].astype('int') # 支路起始节点
    t = branch[:, 1].astype('int') # 支路末端节点

    ij = list(zip(f,t)) # 线路集合
    r = branch[:,2]/(basekV**2/baseMVA) # 电阻有名值化为标幺值
    x = branch[:,3]/(basekV**2/baseMVA) #  电抗有名值化为标幺值
    r_ij = dict(zip(ij,r)) # 将电阻与支路对应
    x_ij = dict(zip(ij,x)) # 将电抗与支路对应
    upStream = {Node:branch[branch[:,1]==Node][:,0].astype('int') for Node in bus_num} # 所有节点的上游节点
    downStream = {Node:branch[branch[:,0]==Node][:,1].astype('int') for Node in bus_num} # 所有节点的下游节点

    ### 创建模型和变量

    #%% 建立模型
    model = Model('DistFlow')
    model.setParam("LogToConsole",0)
    GP_i = model.addVars(gen_num,lb=-GRB.INFINITY,name='GP_i') # 发电机有功出力
    GQ_i = model.addVars(gen_num,lb=-GRB.INFINITY,name='GQ_i') # 发电机无功出力
    P_ij = model.addVars(ij, lb=-GRB.INFINITY,name='P_ij') # 线路无功潮流
    Q_ij = model.addVars(ij, lb=-GRB.INFINITY,name='Q_ij') # 线路有功潮流
    l_ij = model.addVars(ij, lb=-GRB.INFINITY,ub=I_max,name='l_ij') # 线路电流
    v_i = model.addVars(bus_num, lb=V_min,ub=V_max,name='v_i') # 节点电压

    ### 功率平衡约束

    #%% 功率平衡约束
    # 非发电机节点功率平衡
    model.addConstrs((0==P_i[i]+quicksum(P_ij[i,j] 
        for j in downStream[i])-quicksum(P_ij[k,i]-r_ij[k,i]*l_ij[k,i] for k in upStream[i]) 
        for i in commen_num),name='NodePBalance')
    model.addConstrs((0==Q_i[i]+quicksum(Q_ij[i,j] 
        for j in downStream[i])-quicksum(Q_ij[k,i]-x_ij[k,i]*l_ij[k,i] for k in upStream[i]) 
        for i in commen_num),name='NodeQBalance')

    # 发电机节点功率平衡
    model.addConstrs((0==-GP_i[i]+P_i[i]+quicksum(P_ij[i,j] 
        for j in downStream[i])-quicksum(P_ij[k,i]-r_ij[k,i]*l_ij[k,i] for k in upStream[i]) 
        for i in gen_num),name='NodeGPBalance')
    model.addConstrs((0==-GQ_i[i]+Q_i[i]+quicksum(Q_ij[i,j] 
        for j in downStream[i])-quicksum(Q_ij[k,i]-x_ij[k,i]*l_ij[k,i] for k in upStream[i]) 
        for i in gen_num),name='NodeGQBalance')

    ### 发电机出力上下限
    model.addConstrs((GP_i[i]<=genP_max[i] for i in gen_num),name='Pmax') # 发电机有功出力上限
    model.addConstrs((GP_i[i]>=-genP_max[i] for i in gen_num),name='Pmin') # 发电机有功出力下限
    model.addConstrs((GQ_i[i]<=genQ_max[i] for i in gen_num),name='Qmax') # 发电机无功出力上限
    model.addConstrs((GQ_i[i]>=-genQ_max[i] for i in gen_num),name='Qmin') # 发电机无功出力下限

    ### 二阶锥约束
    # 修改后的约束  11.1
    model.addConstrs((  (2*P_ij[i,j])**2 + (2*Q_ij[i,j])**2 + (v_i[i] - l_ij[i,j])**2 <= ( (v_i[i]+ l_ij[i,j])**2) for i,j in ij),name='SOC')

    ### 电压电流约束
    model.addConstrs((v_i[j]==v_i[i]-2*(r_ij[i,j]*P_ij[i,j]+x_ij[i,j]*Q_ij[i,j])+(r_ij[i,j]**2+x_ij[i,j]**2)*l_ij[i,j] for (i,j) in ij),name='voltage')
    model.addConstrs((v_i[i]>=V_min for i in bus_num),name='voltageMin')
    model.addConstrs((v_i[i]<=V_max for i in bus_num),name='voltageMax')
    #model.addConstr((v_i[31]==1),name='slackNode')
    model.addConstr((v_i[1]==1),name='slackNode') # 1025
    model.addConstrs((l_ij[i,j]<=I_max for i,j in ij),name='Lijconstrs')
    #model.addConstr((v_i[31]==1),name='slackNode')
    model.addConstr((v_i[1]==1),name='slackNode') # 1025
    model.addConstrs((l_ij[i,j]<=I_max for i,j in ij),name='Lijconstrs')
    
    # update约束
    model.update()

    ### 定义目标函数及模型求解

    obj = quicksum(GP_i[i]**2 for i in gen_num)
    model.setObjective(obj,GRB.MINIMIZE)
    model.optimize()

    X = np.zeros([busNum,busNum])
    R = np.zeros([busNum,busNum])
    V = np.array([[v_i[i].X for i in v_i.keys()]])
    IL = np.array([l_ij[i].X for i in l_ij.keys()])

    for (ix,jx),vx in x_ij.items():
        X[ix-1,jx-1] = vx

    for (ir,jr),vr in r_ij.items():
        R[ir-1,jr-1] = vr

    Pg = GP_i[1].X

    IL = np.hstack([branch[:,:2],IL.reshape(-1,1)])
    return V, IL,  R, X, Pg