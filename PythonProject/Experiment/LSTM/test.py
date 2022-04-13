import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # load the dataset
    dataframe = pd.read_excel('data.xlsx', usecols=[0])
    # print(dataframe)
    print("数据集的长度：", len(dataframe))
    dataset = dataframe.values
    # 将整型变为float
    dataset = dataset.astype('float32')

    plt.plot(dataset)
    plt.show()

    # 数据格式转换为监督学习，归一化数据，训练集和测试集划分
    # X是给定时间(t)的乘客人数，Y是下一次(t + 1)的乘客人数。
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

    # fix random seed for reproducibility
    numpy.random.seed(7)

    # 数据缩放
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 将数据拆分成训练和测试，2/3作为训练数据
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print("原始训练集的长度：", train_size)
    print("原始测试集的长度：", test_size)

    # 构建监督学习型数据

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    print("转为监督学习，训练集数据长度：", len(trainX))
    # print(trainX,trainY)
    print("转为监督学习，测试集数据长度：", len(testX))
    # print(testX, testY )
    # 数据重构为3D [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print('构造得到模型的输入数据(训练数据已有标签trainY): ', trainX.shape, testX.shape)

    # from attention import Attention

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    # Attention(name='attention_weight')
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=80, batch_size=1, verbose=2)

    # 打印模型
    model.summary()

    # 开始预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # 逆缩放预测值
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # 计算误差
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    # 预测下一个月数据
    # 预测未来的数据

    # 测试数据的最后一个数据没有预测,这里补上
    finalX = numpy.reshape(test[-1], (1, 1, testX.shape[1]))

    # 预测得到标准化数据
    featruePredict = model.predict(finalX)

    # 将标准化数据转换为人数
    featruePredict = scaler.inverse_transform(featruePredict)

    # 原始数据是1949-1960年的数据,下一个月是1961年1月份
    print('下一个轨迹点的x坐标 ', featruePredict)
