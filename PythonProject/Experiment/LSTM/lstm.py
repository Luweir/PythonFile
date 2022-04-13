import numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def load_data(dataframe):
    # dataframe = pd.read_excel('data.xlsx', usecols=[0])
    # print(dataframe)
    print("数据集的长度：", len(dataframe))
    dataset = dataframe.values
    # 将整型变为float
    dataset = dataset.astype('float32')
    plt.plot(dataset)
    plt.show()
    return dataset


# 数据格式转换为监督学习，归一化数据，训练集和测试集划分
# 将值数组转换为数据集矩阵,look_back是步长。
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        # X按照顺序取值
        dataX.append(a)
        # Y向后移动一位取值
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def model_run(train_x, train_y, look_back=1):
    print("转为监督学习，训练集数据长度：", len(train_x))
    # 数据重构为3D [samples, time steps, features]
    train_x = numpy.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    print('构造得到模型的输入数据(训练数据已有标签train_y): ', train_x.shape)
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    # Attention(name='attention_weight')
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, epochs=80, batch_size=1, verbose=2)
    # 打印模型
    model.summary()
    return model


def predict(model, look_back, train_x, train_y, test_x, test_y):
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    # 逆缩放预测值
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])

    # 计算误差
    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % (train_score))
    test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (test_score))

    # shift train predictions for plotting
    train_predictPlot = numpy.empty_like(dataset)
    train_predictPlot[:, :] = numpy.nan
    train_predictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    # shift test predictions for plotting
    test_predictPlot = numpy.empty_like(dataset)
    test_predictPlot[:, :] = numpy.nan
    test_predictPlot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_predict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(train_predictPlot)
    plt.plot(test_predictPlot)
    plt.show()


if __name__ == '__main__':
    data = pd.read_excel("data.xlsx", usecols=[1])
    dataset = data.values[:40]
    look_back = 1
    # 一、加载数据集
    # dataset = load_data(data)
    # 数据缩放
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # 将数据拆分成训练和测试   1/2作为训练数据
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print("原始训练集的长度：", train_size)
    print("原始测试集的长度：", test_size)

    # 构建监督学习型数据
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)
    print("转为监督学习，训练集数据长度：", len(train_x))
    print("转为监督学习，测试集数据长度：", len(test_y))

    # 二、模型训练, 得到模型结果
    model = model_run(train_x, train_y, look_back)
    # 三、开始预测
    predict(model, look_back, train_x, train_y, test_x, test_y)
