import threading
import time


class myThread(threading.Thread):
    def __init__(self, threadID, begin, end):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.begin = begin
        self.end = end

    def run(self):
        count = 0
        for i in range(self.begin, self.end):
            count += i
        print("Thread-", self.threadID, ":", count)


if __name__ == '__main__':
    threads = []
    start = 1
    end = 1000000000
    start_time = time.perf_counter()
    while start < end:
        # 创建新线程
        thread1 = myThread(int(start / 5000000), start, start + 5000000)
        # 开启新线程
        thread1.start()
        # 添加线程到线程列表
        threads.append(thread1)
        start += 5000000

    # 等待所有线程完成
    for t in threads:
        t.join()
    end_time = time.perf_counter()
    print("多线程时间：", end_time - start_time)
    start = 1
    count = 0
    start_time = time.perf_counter()
    for i in range(start, end + 1):
        count += i
    end_time = time.perf_counter()
    print("单线程时间：", end_time - start_time)
