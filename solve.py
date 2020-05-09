# -*- encoding: utf-8 -*-"
import time
import copy
import numpy as np
import pandas as pd

time.time()
time_start = time.time()

MAX_DEPTH = 7  # 最大搜索深度，若搜14，则取14/2=7
BLOCK_SIZE = int(64)  # ldpc块大小

# 数据的读取
path = './Example.csv'
people_relation = pd.read_csv(path, sep=',', header=None).values
shape = people_relation.shape
if shape[0] == shape[1] == 1344:
    BLOCK_SIZE = int(192)  # ldpc块大小
elif shape[0] == 256 and shape[1] == 640:
    BLOCK_SIZE = int(64)  # ldpc块大小
    people_relation = people_relation.T
    shape = people_relation.shape

# 创建used矩阵，用于记录某点是否用过
used = np.zeros((shape[0], shape[1]), dtype=int)  # 模拟刻名字，为1表示已经刻过了

# 创建邻接表，这里直接用矩阵表示，其中值为-1表示结束
row_relation = np.full((shape[0], int(shape[1] / BLOCK_SIZE)), -1, dtype=int)  # row_relation[i,:]表示i行上有1的地方的列坐标的的全体
col_relation = np.full((int(shape[0] / BLOCK_SIZE), shape[1]), -1, dtype=int)  # col_relation[:,i]表示i列上有1的地方的行坐标的全体
# TODO:行和列的最大和自动确定
for i in range(shape[0]):
    index = 0
    for j in range(shape[1]):
        if people_relation[i][j] == 1:
            row_relation[i][index] = j
            index += 1
for j in range(shape[1]):
    index = 0
    for i in range(shape[0]):
        if people_relation[i][j] == 1:
            col_relation[index][j] = i
            index += 1

# 初始搜索方向
init_search_direction = 0  # 为0表示最初是按行搜索的，为1表示最初是按列搜索的
track_nums_0 = np.zeros((shape[0], shape[1], 6), dtype=int)  # 最初按行搜索的路径数矩阵

stack_depth = 0  # 当前搜索的栈深
ans = {4: 0, 6: 0, 8: 0, 10: 0, 12: 0, 14: 0}  # 存最终答案的字典
track_list = []  # 用于维护搜索时的某一条路径
track_dict = {}  # 用于存搜到的储路径进行去重  key:(当前行，当前列，当前位置第几次到达, 到达时的栈深) value:每次搜索到的track_list
solve_list = [set() for i in range(6)]  # 维护一个已有解的矩阵，用于存储从每个节点出发的所有可能路径，用于去重，当此节点搜索完成后清空

duplicate_set = {}  # 每一个block(子矩阵)中的重复数
duplicate_count = [np.zeros((BLOCK_SIZE - 1,), dtype=int) for i in range(6)]  # 计算在BLOCK_SIZE块中有多少个重复
duplicate_weight = np.arange(BLOCK_SIZE - 1, 0, -1)  # 每个重复所占权重


def DFS(row, col, direction):
    '''
    row表示当前搜索坐标第row行，col表示当前搜索坐标第col列
    direction: 值为0, 1分别表示按行搜索和按列搜索
    '''
    global g_i, g_j, stack_depth
    if (row == g_i or col == g_j) and stack_depth >= 2:
        return
    if stack_depth in {2, 3, 4, 5, 6, 7}:  # 搜索14个名字的情况，栈深为14
        # 起始为横向的情况：路径数矩阵+1，并将路径存入字典
        if init_search_direction == 0:
            track_nums_0[row][col][stack_depth - 2] += 1  # 到达次数+1
            track_dict[(row, col, track_nums_0[row][col][stack_depth - 2], stack_depth - 2)] = copy.copy(track_list)
        # 起始为纵向的情况：一一进行配对，并存入结果集合
        else:
            track_nums_tmp = track_nums_0[row][col][stack_depth - 2]
            start_pos = [g_i, g_j + shape[0]] + track_list
            while track_nums_tmp:  # 若这里有路径，进行配对
                track = start_pos + track_dict[(row, col, track_nums_tmp, stack_depth - 2)] # 搜索路径
                # 排序 TODO:现在是用set去重+list排序，或许可以自己实现排序，以优化效率
                solve = list(set(track))  # 利用set去除无需短环
                # 只有不是无效短环的才需要对其进行排序，优化性能
                if len(solve) == stack_depth * 2:
                    solve.sort()
                    # 加入结果集合中，由于list和set都是unhashable，因此需要转为tuple
                    solve = tuple(solve)
                    # 避免重复，先判断是否在里面
                    if solve not in solve_list[stack_depth - 2]:
                        solve_list[stack_depth - 2].add(tuple(solve))
                        duplicate_set = set(range(g_i + 1, g_i + BLOCK_SIZE)) & set(solve)
                        # 如果当前搜索到的这条路径中，存在经过本块内部的点，需要去重
                        if duplicate_set:
                            duplicate_count[stack_depth - 2][(BLOCK_SIZE - max(duplicate_set) - 1) % BLOCK_SIZE] += 1
                track_nums_tmp -= 1
        if stack_depth == MAX_DEPTH:  # 到最大深度返回
            return
    stack_depth += 1
    if direction == 0:  # 在本行搜索
        for i in row_relation[row]:
            if i != -1 and i != col and used[row][i] == 0 and (i != g_j or (row == g_i and i == g_j)):  # 搜索限制：1.是好友，即关系值为1 2.要把祭品给对方部落，即和上次搜索行号不同 3.此人没有刻过名字，即该块未用过 4.不能在初始列上，除非回到起始元素或起始是按行搜索的（这个找不到对应的实际条件，目的仅仅是去重）
                used[row][i] = 1
                track_list.append(i+shape[0])
                DFS(row, i, 1)
                track_list.pop()
                used[row][i] = 0
    elif direction == 1:  # 在本列搜索
        for i in col_relation[:, col]:
            if i != -1 and i != row and used[i][col] == 0 and (i != g_i or (col == g_j and i == g_i)):  # 搜索限制：1.是好友，即关系值为1 2.要把祭品给对方部落，即和上次搜索列号不同 3.此人没有刻过名字，该块未用过
                used[i][col] = 1
                track_list.append(i)
                DFS(i, col, 0)
                track_list.pop()
                used[i][col] = 0
    stack_depth -= 1
    return  # 无法满足条件，不再传递该祭品


if __name__ == '__main__':
    for g_i in range(0, shape[0], BLOCK_SIZE):
        print('current row：', g_i)
        for g_j in row_relation[g_i]:
            if g_j != -1:
                print('    current col：', g_j)
                # 横向搜7步
                init_search_direction = 0
                DFS(g_i, g_j, 0)
                # 纵向搜7步
                init_search_direction = 1
                DFS(g_i, g_j, 1)

                # 记录已经搜索完的节点
                track_nums_0 = np.zeros((shape[0], shape[1], 6), dtype=int)  # 最初按行搜索的路径数矩阵
                row_relation[g_i][np.where([row_relation[g_i] == g_j])[1]] = -1
                col_relation[np.where([col_relation[:, g_j] == g_i])[1][0]][g_j] = -1

                # 下面是一些全局变量的重置
                track_dict.clear()  # 用于存储路径进行去重  key:(当前行，当前列，当前位置第几次到达) value:每次搜索到的track_list

        # 记录已经搜索完的节点
        row_relation[g_i:g_i + BLOCK_SIZE, :] = np.full((BLOCK_SIZE, int(shape[1] / BLOCK_SIZE)), -1, dtype=int)
        for i in range(int(shape[0] / BLOCK_SIZE)):  # TODO:后续要改掉，不用循环写
            for j in range(shape[1]):
                if g_i <= col_relation[i][j] < g_i + BLOCK_SIZE:
                    col_relation[i][j] = -1

        # 去除无效短环和由LDPC码特点引入的重复
        for depth in range(6):  # 每一个搜索深度的情况逐个处理
            ans[(depth + 2) * 2] += BLOCK_SIZE * len(solve_list[depth]) - duplicate_count[depth].dot(duplicate_weight)  # 减去重复的和

        # 全局变量的重置
        duplicate_count = [np.zeros((BLOCK_SIZE - 1,), dtype=int) for i in range(6)]
        solve_list = [set() for i in range(6)]

    print('========search finished========')

    # 输出运行时间
    time_end = time.time()
    print('time cost: ', time_end - time_start)

    # 输出结果到TERMINAL
    print('num_4: ' + str(ans[4]))
    print('num_6: ' + str(ans[6]))
    print('num_8: ' + str(ans[8]))
    print('num_10: ' + str(ans[10]))
    print('num_12: ' + str(ans[12]))
    print('num_14: ' + str(ans[14]))

    # 写入到文件中
    f = open('./result.txt', 'w')
    f.write(str(ans[4]) + '\r\n')
    f.write(str(ans[6]) + '\r\n')
    f.write(str(ans[8]) + '\r\n')
    f.write(str(ans[10]) + '\r\n')
    f.write(str(ans[12]) + '\r\n')
    f.write(str(ans[14]) + '\r\n')
    f.close()