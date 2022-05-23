import matplotlib.pyplot as plt

# 单线图轨迹位置，传入DataFrame data
def get_line_chart(data):
    plt.figure(figsize=(12, 8))
    # plt的线图则会按输入顺序来
    plt.plot(data['longitude'], data['latitude'])
    plt.show()


# 双线图比较轨迹位置，传入DataFrame data1和data2
def get_two_line_chart(data1, data2):
    plt.figure(figsize=(12, 8))
    plt.plot(data1['longitude'], data1['latitude'], color='red')  # 第一条红色
    plt.plot(data2['longitude'], data2['latitude'], color='blue')  # 第二条蓝色
    plt.show()
