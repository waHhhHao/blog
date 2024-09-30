# @Time: 2024/8/24 17:16
# @Author: xy
import copy

from utils import *
from collections import defaultdict


def viterbi(observation_list, E, T, I=None):
    """
    对于观测：我是谁, 最优观测路径是BME
    若最优状态路径为BME, 那么第二个状态为M的所有路径中概率最大的一定是BM

    最大状态概率路径: 是指走到时间步t时刻, 分别以B,E,M,S为最后状态的路径中概率最大的那条路, 即max(*B), max(*E), max(*M), max(*S)
    最大状态概率路径的更新, 即 max(*B)_{t+1} = max(max_{t}(*B)*T(BB)E(O|B),  max_{t}(*E)*T(EB)E(O|B), max_{t}(*M)*T(MB)E(O|B), max_{t}(*S)*T(SB)E(O|B))
    :param observation_list: 观测列表
    :param E: 发射矩阵
    :param T: 转移矩阵
    :param I: 初始概率分布
    :return:
    """
    # 第一个observation的状态分布函数, 也即时间步t=0时, 处于各个状态的(最大)概率
    I = {
        "B": 1 / 2,
        "M": 0,
        "E": 0,
        "S": 1 / 2,
    }

    status_set = I.keys()
    path = {status: [[], p] for status, p in I.items()}  # 存储以status结尾的最大概率路径以及概率

    # 初始化t=0时刻的path
    for status in status_set:
        p = I[status]
        path[status][0] = [status]
        path[status][1] = p
    print(f"时刻0下的最大概率路径：{path}")

    for t, observation in enumerate(observation_list[1:], start=1):
        new_path_t = copy.deepcopy(path)
        # 根据计算时间步t下的观察, 计算上一个状态哪个值的概率最大
        for current_status in status_set:
            # 计算当前步下以current_status为结束状态的最大概率路径和对应概率
            max_p, max_prev_status = max(
                (path[prev_status][1] * T[prev_status][current_status] * E[current_status][observation], prev_status)
                for prev_status in status_set
            )
            new_path_t[current_status][0] = path[max_prev_status][0] + [current_status]
            new_path_t[current_status][1] = max_p
        path = new_path_t
    return path


if __name__ == "__main__":
    E = load_json(r"./matrices/emission_matrix.json")
    T = load_json(r"./matrices/transition_matrix.json")
    observations = list("爱中国")
    path = viterbi(observations, E, T)
    print(path)
