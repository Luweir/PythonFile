import Experiment.FastDTW.fast_dtw_neigbors as dtw

if __name__ == '__main__':
    print(dtw.get_fastdtw([[4, 0], [5, 0.25], [7, 2.25], [8, 4], [11, 12.25], [14, 25]],
                          [[0, 0], [2, 1], [4, 4], [6, 9], [8, 16], [10, 25]]))
