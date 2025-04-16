def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2):
    """
    :param actual_lst: 真实的序列值
    :param pred1_lst: 第一个模型预测的结果
    :param pred2_lst: 第二个模型预测的结果
    :param h: 预测模型是几步预测，h就是几
    :param crit: 计算连个模型的预测偏差，的差值 d 时，使用的公式：有MSE,MAD,MAPE,poly，推荐MSE
    :param power: 只有crit=poly是用这个，计算d时使用： (模型1的偏差)的power次方 - (模型2的偏差)的power次方
    """
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if not isinstance(h, int):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return rt, msg
        # Check the range of h
        if h < 1:
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return rt, msg
        len_act = len(actual_lst)
        len_p1 = len(pred1_lst)
        len_p2 = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2:
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return rt, msg
        # Check range of h
        if h >= len_act:
            rt = -1
            msg = "The number of steps ahead is too large."
            return rt, msg
        # Check if criterion supported
        if crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly":
            rt = -1
            msg = "The criterion is not supported."
            return rt, msg
            # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")

        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True

        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if not (is_actual_ok and is_pred1_ok and is_pred2_ok):
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return rt, msg
        return rt, msg

    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if error_code[0] == -1:
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np

    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst = []

    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()

    # Length of lists (as real numbers)
    T = float(len(actual_lst))

    # construct d according to crit
    if crit == "MSE":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append((actual - p1) ** 2)
            e2_lst.append((actual - p2) ** 2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif crit == "MAD":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif crit == "MAPE":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs((actual - p1) / actual))
            e2_lst.append(abs((actual - p2) / actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif crit == "poly":
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(((actual - p1)) ** (power))
            e2_lst.append(((actual - p2)) ** (power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)

            # Mean of d
    mean_d = pd.Series(d_lst).mean()

    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N - k):
            autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
        return (1 / (T)) * autoCov

    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    DM_stat = V_d ** (-0.5) * mean_d
    harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** (0.5)
    DM_stat = harvey_adj * DM_stat
    # Find p-value
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    rt = dm_return(DM=float(DM_stat), p_value=float(p_value))
    return rt

import pandas as pd
def main():
    #data = pd.read_excel(r"E:\bin\DM检验.xlsx")
    #actual_lst = data['真实值']
    #pred1_lst = data['模型1预测值']
    #pred2_lst = data["模型2预测值"]
    actual_lst = [92.62, 98, 94.67, 99, 72.47, 56, 66.21, 59, 57.27, 58.5]
    pred1_lst = [88.665, 96.85, 87.307, 99.109, 66.863, 53.377, 64.687, 54.09, 56.259, 54.193]
    pred2_lst = [91.951, 90.47, 100.585, 91.154, 73.012, 64.346, 71.155, 63.063, 56.914, 58.147]

    pred3_lst = actual_lst  # 这里假设模型3完美预测

    print("result")
    rt = dm_test(actual_lst, pred1_lst, pred2_lst, crit="MAPE")
    print(rt)

    print("-------______下面是使用不同的crit得到的结果____-------")
    rt = dm_test(actual_lst, pred1_lst, pred2_lst, crit="MSE")
    print("crit:MSE,", rt)

    rt = dm_test(actual_lst, pred1_lst, pred2_lst, crit="poly", power=4)
    print("crit: poly 4,", rt)

if __name__ == '__main__':
    main()

