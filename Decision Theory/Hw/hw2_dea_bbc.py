from gurobipy import *
import pandas as pd
import numpy as np


class DEA(object):
    def __init__(self, DMU_name, X, Y):
        self.m1 = X.shape[1]
        self.m2 = Y.shape[1]
        self.DMUs, self.X, self.Y = multidict({DMU: [X.loc[DMU].tolist(),
                                                     Y.loc[DMU].tolist()] for DMU in DMU_name})

    def ccr(self):
        """计算theta dd 的值"""
        lst = []
        a = []
        for d in self.DMUs:
            m = Model()
            v = m.addVars(self.m1)
            u = m.addVars(self.m2)
            m.update()

            m.setObjective(quicksum(u[i] * self.Y[d][i] for i in range(self.m2)), sense=GRB.MAXIMIZE)
            m.addConstr(quicksum(v[i] * self.X[d][i] for i in range(self.m1)) == 1)
            m.addConstrs(quicksum(u[i] * self.Y[j][i] for i in range(self.m2))
                         - quicksum(v[i] * self.X[j][i] for i in range(self.m1)) <= 0 for j in self.DMUs)
            m.setParam('OutputFlag', 0)
            m.setParam('NonConvex', 2)
            m.optimize()
            lst.append(m.objVal)
        return lst


if __name__ == '__main__':
    data = pd.read_excel('C:/Users/wzu/Desktop/例1.xlsx', index_col=0, header=0)
    i1 = 2  # 表示表格中X有几列
    i2 = 1
    X = data[data.columns[:i1]]  # 原始数据
    Y = data[data.columns[i1:i1 + i2]]
    dea = DEA(DMU_name=data.index, X=X, Y=Y)
    print(dea.ccr()[0])


    def ccr(self):
        for k in self.DMUs:
            m = Model()
            theta = m.addVar()  # 单个变量不加s
            lambdas = m.addVars(self.DMUs)
            m.update()  # 以上都是变量
            m.setObjective(theta, sense=GRB.MAXIMIZE)
            # 添加约束
            m.addConstrs(self.X[k][j] >= quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs) for j in range(self.m1))
            m.addConstrs(
                theta * self.Y[k][j] <= quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs) for j in range(self.m2))

            # setParam(paramname,newvalue)是修改Gurobi参数，使运行结果更快，参考第三章P75
            m.setParam('OutputFlag', 0)  # 去掉计算过程，只取最后结果;
            # m.setParam('NonConvex', 2) #凸与非凸再查查
            m.optimize()
            print(m.objVal)

    def ccr2(self):
        for k in self.DMUs:
            lst = []
            m = Model()
            theta = m.addVar()
            lambdas = m.addVars(self.DMUs)
            sx = m.addVars(self.m1)
            sy = m.addVars(self.m2)
            m.update()
            m.setObjective(theta, sense=GRB.MINIMIZE)
            # 添加约束
            m.addConstrs(theta*self.X[k][j] - sx[j] == quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs) for j in range(self.m1))
            m.addConstrs(self.Y[k][j] + sy[j] == quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs) for j in range(self.m2))
            m.setParam('OutputFlag', 0)
            m.setParam('NonConvex', 2)
            m.optimize()
            for j in range(self.m1):
                lst.append(sx[j].x)
            for j in range(self.m2):
                lst.append(sy[j].x)
            print(m.objVal)
            print(lst) #导出的为乘数

    def ccr3(self):
        for k in self.DMUs:
            m = Model()
            theta = m.addVar()
            lambdas = m.addVars(self.DMUs)
            sx = m.addVars(self.m1)
            sy = m.addVars(self.m2)
            m.update()
            m.setObjective(theta, sense=GRB.MAXIMIZE)
            #添加约束
            m.addConstrs(self.X[k][j] - sx[j] == quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs) for j in range(self.m1))
            m.addConstrs(theta*self.Y[k][j] + sy[j] == quicksum(lambdas[i]* self.Y[i][j] for i in self.DMUs) for j in range(self.m2))
            m.setParam('OutputFlag', 0)
            #m.setParam('NonConvex', 2)
            m.optimize()
            print(m.objVal)


    def bcc(self):
        for k in self.DMUs:
            m = Model()
            theta = m.addVar()  # 单个变量不加s
            lambdas = m.addVars(self.DMUs)
            m.update()  # 以上都是变量
            m.setObjective(theta, sense=GRB.MAXIMIZE)
            # 添加约束
            m.addConstrs(self.X[k][j] >= quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs) for j in range(self.m1))
            m.addConstrs(
                theta * self.Y[k][j] <= quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs) for j in range(self.m2))
            m.addConstr(quicksum(lambdas[i] for i in self.DMUs) == 1)

            # setParam(paramname,newvalue)是修改Gurobi参数，使运行结果更快，参考第三章P75
            m.setParam('OutputFlag', 0)  # 去掉计算过程，只取最后结果;
            # m.setParam('NonConvex', 2) #凸与非凸再查查
            m.optimize()
            print(m.objVal)

    def SBM(self):
        for k in self.DMUs:
            lst = []
            m = Model()
            t = m.addVar()
            lambdas = m.addVars(self.DMUs)
            Sx = m.addVars(self.m1)
            Sy = m.addVars(self.m2)
            m.update()
            m.setObjective(t - 1/self.m1 * quicksum(Sx[j] / self.X[k][j] for j in range(self.m1)), sense=GRB.MINIMIZE)
            # 添加约束
            m.addConstrs(
                t * self.X[k][j] - Sx[j] == quicksum(lambdas[i] * self.X[i][j] for i in self.DMUs) for j in range(self.m1))
            m.addConstrs(
                t * self.Y[k][j] + Sy[j] == quicksum(lambdas[i] * self.Y[i][j] for i in self.DMUs) for j in range(self.m2))
            m.addConstr(t + 1/self.m2*quicksum(Sy[j] / self.Y[k][j] for j in range(self.m2)) == 1)
            m.setParam('OutputFlag', 0)
            m.setParam('NonConvex', 2)
            m.optimize()
            print(m.objVal)
            for j in range(self.m1):
                lst.append(Sx[j].x)
            for j in range(self.m2):
                lst.append(Sy[j].x)
            print(lst)