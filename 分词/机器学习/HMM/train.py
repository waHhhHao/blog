# @Time: 2024/8/24 17:15
# @Author: xy
from collections import defaultdict, Counter
import ast
from utils import *

"""
计算转移矩阵和发射矩阵, 存为json, 放在./matrices下面
"""


def get_transition_matrix(states):
    """
    计算状态转移矩阵, p(s1->s2) = count(s1->s2)/count(s1)
    :param states:
    :return:
    """
    # 初始化转移次数的计数器
    transition_counts = defaultdict(Counter)

    # 初始化状态出现次数的计数器
    state_counts = Counter()

    # 计算转移次数
    for i in range(len(states)):
        current_state = states[i]
        next_state = 'S' if i == len(states) - 1 else states[i + 1]
        transition_counts[current_state][next_state] += 1
        state_counts[current_state] += 1

    # 构建状态转移矩阵
    transition_matrix = {state: {state: 0 for state in 'BEMS'} for state in 'BEMS'}

    # 计算转移概率
    for current_state, transitions in transition_counts.items():
        total = state_counts[current_state]
        for next_state, count in transitions.items():
            transition_matrix[current_state][next_state] = count / total

    return transition_matrix


def get_emission_matrix(pairs):
    """
    计算发射矩阵 p(s->w) = count(s->w)|count(s)
    :return:
    """
    emission_matrix = defaultdict(Counter)
    total_count = Counter()  # 记录某个状态出现的次数
    for pair in pairs:
        word = pair[0]
        status = pair[1]
        emission_matrix[status][word] += 1
        total_count[status] += 1
    for status, counter in emission_matrix.items():
        for word, cnt in counter.items():
            emission_matrix[status][word] = cnt / total_count[status]
    return emission_matrix


if __name__ == "__main__":
    word_status_pair_list = []
    with open(r"./ok_corpus/train.txt", "r", encoding='utf-8') as data:
        for line in data:
            word_status_pair_list.append(ast.literal_eval(line))
    # 计算状态转移矩阵
    states = [p[1] for p in word_status_pair_list]
    transition_matrix = get_transition_matrix(states=states)
    save_json(r"./matrices/transition_matrix.json", transition_matrix)
    # 计算发射矩阵
    emission_matrix = get_emission_matrix(word_status_pair_list)
    save_json(r"./matrices/emission_matrix.json", emission_matrix)
