import time
from Experiment.Threshold.threshold import gps_reader, get_out
from Experiment.DP.dp import get_sed, get_ped


def opw(points, epsilon):
    original_index = 0
    result_index = [original_index]
    e = original_index + 2
    while e < len(points):
        i = original_index + 1
        cond_opw = True
        while i < e and cond_opw:
            if get_ped(points[original_index], points[i], points[e]) > epsilon:
                cond_opw = False
            else:
                i += 1
        if not cond_opw:
            original_index = i
            result_index.append(original_index)
            e = original_index + 2
        else:
            e += 1
    result_index.append(len(points) - 1)
    return result_index


def opw_tr(points, epsilon):
    original_index = 0
    result_index = [original_index]
    e = original_index + 2
    while e < len(points):
        i = original_index + 1
        cond_opw = True
        while i < e and cond_opw:
            if get_sed(points[original_index], points[i], points[e]) > epsilon:
                cond_opw = False
            else:
                i += 1
        if not cond_opw:
            original_index = i
            result_index.append(original_index)
            e = original_index + 2
        else:
            e += 1
    result_index.append(len(points) - 1)
    return result_index


if __name__ == '__main__':
    filename = r"../data/20081023025304-0.plt"
    epsilon = 0.0001
    save_filename = "result.csv"
    points = gps_reader(filename)
    start_time = time.perf_counter()
    sample_index = opw_tr(points, epsilon)
    sample = []
    for e in sample_index:
        sample.append(points[e])
    end_time = time.perf_counter()
    print("opw")
    get_out(save_filename, start_time, end_time, points, sample)
