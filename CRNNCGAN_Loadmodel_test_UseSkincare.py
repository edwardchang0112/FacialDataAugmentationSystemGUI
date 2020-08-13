import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
import datetime
import csv
#from OpenWeatherAPI import get_future_weather
import random

def readcsv(file_name, factor):
    dataset = pd.read_csv(file_name)
    #print (dataset[factor])
    return dataset[factor]

def readExcel(file_name, factor):
    dataset = pd.read_excel(file_name)
    #print (dataset[factor])
    return dataset[factor]

def store_csv(Augment_Skin_data, window_size, file_name_No):
    with open('Generated_User_Data/'+str(file_name_No)+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Date', 'Date_Diff', 'Temperature', 'Humidity', 'Skincare_Ratio', 'Age', 'Avg3_Hydration', 'Avg3_Skinhealth'])
        writer.writerows(Augment_Skin_data)

def moving_window(ts_data, window_size):
    #ts_data_win = np.zeros((len(ts_data)-window_size+1, window_size))
    ts_data_win = []
    for i in range(len(ts_data)-window_size+1):
        #print ("ts_data[i:i+window_size] = ", ts_data[i:i+window_size])
        ts_data_win.append(ts_data[i:i+window_size])
    return ts_data_win

def data_rolling(ts_data, window_size):
    ts_data_arr = np.asarray(ts_data)
    ts_data_rol = moving_window(ts_data_arr, window_size)
    #ts_data_all_rol = np.stack((ts_hydration_data_rol, ts_temp_data_rol, ts_humid_data_rol), axis=1)
    return ts_data_rol

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))

def generator(z, y, keep_prob, seq_size, n_hidden, g_num_layers, input_size):
    input = tf.concat(axis=2, values=[z, y])
    input=tf.unstack(input,seq_size,1)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(n_hidden), output_keep_prob=keep_prob)for _ in range(g_num_layers)])
    with tf.variable_scope("gen") as gen:
        res, states = tf.nn.static_rnn(lstm_cell, input,dtype=tf.float32)
        weights=tf.Variable(tf.random_normal([n_hidden, input_size]))
        biases=tf.Variable(tf.random_normal([input_size]))
        for i in range(len(res)):
            res[i]=tf.nn.sigmoid(tf.matmul(res[i], weights) + biases)
        g_params=[v for v in tf.global_variables() if v.name.startswith(gen.name)]
    with tf.name_scope("gen_params"):
        for param in g_params:
            variable_summaries(param)
    return res, g_params

def discriminator(x, x_generated, keep_prob, seq_size, n_hidden, d_num_layers, batch_size):
    #input = tf.concat(axis=2, values=[x, y])
    input=tf.unstack(x, seq_size, 1)
    x_generated=list(x_generated)
    x_in = tf.concat([input, x_generated], 1)
    x_in=tf.unstack(x_in, seq_size, 0)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(n_hidden), output_keep_prob=keep_prob) for _ in range(d_num_layers)])
    with tf.variable_scope("dis") as dis:
        weights=tf.Variable(tf.random_normal([n_hidden, 1]))
        biases=tf.Variable(tf.random_normal([1]))
        outputs, states = tf.nn.static_rnn(lstm_cell, x_in, dtype=tf.float32)
        res=tf.matmul(outputs[-1], weights) + biases
        y_data = tf.nn.tanh(tf.slice(res, [0, 0], [batch_size, -1], name=None))
        y_generated = tf.nn.tanh(tf.slice(res, [batch_size, 0], [-1, -1], name=None))
        d_params=[v for v in tf.global_variables() if v.name.startswith(dis.name)]
    with tf.name_scope("desc_params"):
        for param in d_params:
            variable_summaries(param)
    return y_data, y_generated, d_params


def sample_Z(k, m, n): #生成维度为[m, n]的随机噪声作为生成器G的输入
    #noise_arr_all = []
    #for sample_no in range(k):
    #    noise_arr1 = np.random.normal(0., 0.01, size=[1, n])
    #    noise_arr2 = np.tile(noise_arr1, (m, 1))
    #    noise_arr_all.append(noise_arr2)
    #noise_arr_all = np.vstack((noise_arr_all))
    #noise_arr_all = noise_arr_all.reshape([-1, m, n])
    noise_arr_all = np.random.normal(0., 0.1, size=[k, m, n])
    return noise_arr_all


def xavier_init(size): #初始化参数时使用的xavier_init函数
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.) #初始化标准差
    return tf.random_normal(shape=size, stddev=xavier_stddev) #返回初始化的结果

def save(saver, sess, logdir, step): #保存模型的save函数
    model_name = 'model' #模型名前缀
    checkpoint_path = os.path.join(logdir, model_name) #保存路径
    saver.save(sess, checkpoint_path, global_step=step) #保存模型
    print('The checkpoint has been created.')

def generate_new_user_data(sess, G_sample, model_name, start_file_name_No, n_sample, window_size, Age, date_year, date_month, date_day):
    #start_file_name_No = '001'
    #n_sample = 10
    #Age = 35
    ## For Taiwan
    #logitude = 121
    #latitude = 25

    #future_weather_info = get_future_weather(logitude, latitude)
    #Date_Diff_Generated = np.vstack(([i for i in range(window_size)]))
    #Temperature_Generated = np.vstack(([future_weather_info[i][0] for i in range(window_size)]))
    #Humidity_Generated = np.vstack(([future_weather_info[i][1] for i in range(window_size)]))
    #SkincareRatio_Generated = np.vstack(([random.randint(1, 7) for i in range(window_size)]))
    #Age_Generated = np.vstack(([Age for i in range(window_size)]))

    #date_year = '2020'
    #date_month = '01'
    if sess == None and G_sample == None:
        sess, G_sample = load_trained_CGAN_model(window_size, model_name)
    else:
        pass

    Weather_file_name = 'Weather_info/'+str(date_year)+'-'+str(date_month)+'.xlsx'
    Date_data = readExcel(Weather_file_name, 'ObsTime')
    Date_data_arr = np.vstack((Date_data))
    Date_data_list = []
    start_date_day = int(date_day)-1
    for date_index in range(start_date_day, len(Date_data_arr)):
        if len(str(Date_data_arr[date_index][0])) < 2:
            Date_data_list.append(str(date_year)+'-'+str(date_month)+'-0'+str(Date_data_arr[date_index][0]))
        else:
            Date_data_list.append(str(date_year)+'-'+str(date_month)+'-'+str(Date_data_arr[date_index][0]))

    window_size = min(window_size, len(Date_data_list))
    Date_Generated = np.vstack(([Date_data_list[i] for i in range(window_size)]))
    print("Date_Generated = ", Date_Generated)

    Date_Diff_Generated = np.vstack(([i for i in range(window_size)]))
    Temperature_data = (readExcel(Weather_file_name, 'Temperature')).astype(np.float64)
    Temperature_Generated = np.vstack(([Temperature_data[start_date_day+i] for i in range(window_size)]))
    Humidity_data = (readExcel(Weather_file_name, 'RH')).astype(np.float64)
    Humidity_Generated = np.vstack(([Humidity_data[start_date_day+i] for i in range(window_size)]))
    #SkincareRatio_Generated = np.vstack(([random.randint(1, 7) for i in range(window_size)]))
    #Age_Generated = np.vstack(([Age for i in range(window_size)]))
    #Sample_y_data = np.hstack((Date_Diff_Generated, Temperature_Generated, Humidity_Generated, SkincareRatio_Generated, Age_Generated))
#
    #Sample_y_data = [Sample_y_data[i].reshape(1, -1) for i in range(len(Sample_y_data))]
    #Sample_y_data = np.hstack((Sample_y_data)) / 100
    #print("Sample_y_data = ", Sample_y_data)

    #SkincareRatio = (readcsv(Sample_file_name, 'Skincare_Ratio')).astype(np.float64)
    #Age = (readcsv(Sample_file_name, 'Age')).astype(np.float64)
    #Sample_y_data = np.stack((Date_Diff, Temperature, Humidity, SkincareRatio, Age), axis=1)

    #Sample_y_data = np.stack(
    #    (Date_Diff_Generated, Temperature_Generated, Humidity_Generated, SkincareRatio_Generated, Age_Generated),
    #    axis=1)
    #Sample_y_data = [Sample_y_data[i].reshape(1, -1) for i in range(len(Sample_y_data))]
    #Sample_y_data = np.hstack((Sample_y_data)) / 100
#
    #date_today = datetime.datetime.today()
    #date_today_str = date_today.strftime("%Y/%m/%d")
    #start_date = date_today_str
    #date_today_datefrom = datetime.date(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:10]))
    #print("date_today_datefrom = ", date_today_datefrom)

    #Z = tf.get_collection("Z")
    #y = tf.get_collection("y")
    #print("Z = ", Z)
    #print("y = ", y)

    X_dim = 2  # hydration and oxygen
    y_dim = 5  # day_diff, temperature, humidity, skincareRatio, age
    Z_dim = 10  # 10 dimension of a random input vector
    seq_size_1 = 15
    #y = tf.placeholder(tf.float32, shape=[None, seq_size_1, y_dim], name="y")
    #Z = tf.placeholder(tf.float32, shape=[None, seq_size_1, Z_dim], name="Z")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    keep_prob_value = np.sum(0.7).astype(np.float32)
    seq_size = tf.placeholder(tf.float32, name="seq_size")
    seq_size_value = np.sum(15).astype(np.float32)
    input_size = tf.placeholder(tf.float32, name="input_size")
    input_size_value = np.sum(X_dim).astype(np.float32)
    batch_size = 10
    n_hidden = tf.placeholder(tf.float32, name="n_hidden")
    n_hidden_value = np.sum(30).astype(np.float32)
    g_num_layers = tf.placeholder(tf.float32, name="g_num_layers")
    g_num_layers_value = np.sum(10).astype(np.float32)
    d_num_layers = 10
    #Z_dim = 10
    n_sample_z = 1
    seq_size_z = 15

    file_name = 'GUser_' + start_file_name_No
    user_code_list = []
    user_No_list = []
    for i in range(n_sample):
        date = []
        SkincareRatio_Generated = np.vstack(([random.randint(1, 7) for iter1 in range(window_size)]))
        Age_random = Age + random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        Age_Generated = np.vstack(([Age_random for iter2 in range(window_size)]))
        Sample_y_data = np.hstack(
            (Date_Diff_Generated, Temperature_Generated, Humidity_Generated, SkincareRatio_Generated, Age_Generated))

        Sample_y_data = [Sample_y_data[iter3].reshape(1, -1) for iter3 in range(len(Sample_y_data))]
        Sample_y_data = np.hstack((Sample_y_data)) / 100
        Sample_y_data = Sample_y_data.reshape([-1, 15, 5])
        print("Sample_y_data = ", Sample_y_data)
        #input("======")

        Z_sample = sample_Z(n_sample_z, seq_size_z, Z_dim)
        #Z_sample = sample_Z(1, Z_dim)
        y_sample = Sample_y_data
        samples = sess.run(G_sample, feed_dict={"Z:0": Z_sample, "y:0": y_sample, keep_prob: keep_prob_value, seq_size: seq_size_value, n_hidden: n_hidden_value, g_num_layers: g_num_layers_value, input_size: input_size_value})
        samples = np.asarray(samples).reshape(window_size, -1)
        y_sample = np.asarray(y_sample).reshape(window_size, -1)  # shape (?, 4) for 4 fators in y
        #for i in range(len(y_sample)):
        #    date_shift = date_today_datefrom + datetime.timedelta(days=int(Date_Diff_Generated.flatten()[i]))
        #    date_shift_str = str(
        #        str(date_shift)[:4] + str('-') + str(date_shift)[5:7] + str('-') + str(date_shift)[8:10])
        #    date.append(date_shift_str)
        date = np.vstack((Date_Generated))
        user_data_generated = np.hstack((y_sample, samples))
        user_data_generated = user_data_generated * 100
        date_user_data_generated_header = ['Date', 'Date_Diff', 'Temperature', 'Humidity', 'Skincare_Ratio', 'Age',
                                           'Avg3_Hydration', 'Avg3_Skinhealth']

        data_generated = {
            date_user_data_generated_header[0]: date.flatten(),
            date_user_data_generated_header[1]: user_data_generated[:, 0],
            date_user_data_generated_header[2]: user_data_generated[:, 1],
            date_user_data_generated_header[3]: user_data_generated[:, 2],
            date_user_data_generated_header[4]: user_data_generated[:, 3],
            date_user_data_generated_header[5]: user_data_generated[:, 4],
            date_user_data_generated_header[6]: user_data_generated[:, 5],
            date_user_data_generated_header[7]: user_data_generated[:, 6],
        }

        date_user_data_generated_df = pd.DataFrame(data_generated)
        date_user_data_generated_df.to_excel('Generated_User_Data/' + file_name + '.xlsx', index=None)

        print("Z_sample = ", Z_sample)
        #print("Z_sample[0] = ", Z_sample[0])
        user_code_list.append(Z_sample[0].flatten())
        #print("user_code_list = ", user_code_list)
        #input("==========")
        user_No_list.append(int(file_name[-3:]))

        User_No = str(int(file_name[-3:]) + 1)
        length_zeros = 3 - len(User_No)

        for _ in range(length_zeros):
            User_No = '0' + User_No
        file_name = file_name[:-3] + User_No

    user_code_arr = np.vstack((user_code_list))
    print("user_code_arr = ", user_code_arr)
    create_user_No_code_table(user_No_list, user_code_arr)

    return sess, G_sample

def check_table_exists(table_file_name):
    if os.path.exists(table_file_name):
        print(table_file_name+" exists!")
        return True
    else:
        print(table_file_name+" not exists!")
        return False

def create_user_No_code_table(user_No_list, user_code_list):
    #user_code_table_header = ['User_No', 'User_code_0', 'User_code_1', 'User_code_2', 'User_code_3', 'User_code_4',
    #                          'User_code_5', 'User_code_6', 'User_code_7', 'User_code_8', 'User_code_9']

    user_code_table_header = ['User_No']
    for iter in range(150):
        user_code_table_header.append('User_code_'+str(iter))

    #code_table_generated = {
    #    user_code_table_header[0]: user_No_list,
    #    user_code_table_header[1]: user_code_list[:, 0],
    #    user_code_table_header[2]: user_code_list[:, 1],
    #    user_code_table_header[3]: user_code_list[:, 2],
    #    user_code_table_header[4]: user_code_list[:, 3],
    #    user_code_table_header[5]: user_code_list[:, 4],
    #    user_code_table_header[6]: user_code_list[:, 5],
    #    user_code_table_header[7]: user_code_list[:, 6],
    #    user_code_table_header[8]: user_code_list[:, 7],
    #    user_code_table_header[9]: user_code_list[:, 8],
    #    user_code_table_header[10]: user_code_list[:, 9],
    #}
    code_table_generated = {}
    code_table_generated[user_code_table_header[0]] = user_No_list
    for i in range(1, 151):
        code_table_generated[user_code_table_header[i]] = user_code_list[:, i-1]


    table_file_name = '.User_No_Code_Table.xlsx'
    table_exists = check_table_exists(table_file_name)
    print("code_table_generated = ", code_table_generated)
    date_user_data_generated_df = pd.DataFrame(code_table_generated)
    print("date_user_data_generated_df = ", date_user_data_generated_df)
    if table_exists:
        df = pd.read_excel(table_file_name, index=None)
        #print("code_table_generated.values() = ", np.stack((code_table_generated.values())).T)
        #print("np.stack((df.values)) = ", np.stack((df.values)))
        #print("np.stack((code_table_generated.values())).T = ", np.stack((code_table_generated.values())).T)
        old_df = np.stack((df.values))
        add_df = np.stack((code_table_generated.values())).T
        add_df_copy = add_df.copy()

        new_df = []
        for row_old in old_df:
            duplicate = False
            for row_add in add_df:
                if row_old[0] == row_add[0]:
                    new_df.append(row_add)
                    #print("new_df = ", new_df)
                    print("add_df_copy = ", add_df_copy)
                    print("row_add = ", row_add)
                    ind = np.where(add_df_copy == row_add)
                    add_df_copy = np.delete(add_df_copy, ind[0][0], axis=0)
                    print("add_df_copy = ", add_df_copy)
                    duplicate = True
                    break
            if not duplicate:
                new_df.append(row_old)
            else:
                continue
        new_df.append(add_df_copy)
        new_df = np.vstack((new_df))
        print("new_df = ", new_df)
        #df_1 = np.vstack((np.stack((df.values)), np.stack((code_table_generated.values())).T))
        df_2 = pd.DataFrame(new_df)
        df_2.to_excel(table_file_name, index=None, header=user_code_table_header)
    else:
        date_user_data_generated_df.to_excel(table_file_name, index=None)

def check_date_data_exist(df, start_date):
    if start_date in df.Date.values:
        ind = df[df.Date == start_date].index.tolist()[0]
        df_1 = df[:ind]
        return df_1
    else:
        print("df = ", df)
        return df

def generate_existing_user_data(sess, G_sample, model_name, file_name, window_size, date_year, date_month, date_day):
    #start_file_name_No = '001'
    #n_sample = 10
    #Age = 35
    ## For Taiwan
    #logitude = 121
    #latitude = 25

    if sess == None and G_sample == None:
        sess, G_sample = load_trained_CGAN_model(window_size, model_name)
    else:
        pass
    X_dim = 2  # hydration and oxygen
    y_dim = 5  # day_diff, temperature, humidity, skincareRatio, age
    # y = tf.placeholder(tf.float32, shape=[None, seq_size_1, y_dim], name="y")
    # Z = tf.placeholder(tf.float32, shape=[None, seq_size_1, Z_dim], name="Z")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    keep_prob_value = np.sum(0.7).astype(np.float32)
    seq_size = tf.placeholder(tf.float32, name="seq_size")
    seq_size_value = np.sum(15).astype(np.float32)
    input_size = tf.placeholder(tf.float32, name="input_size")
    input_size_value = np.sum(X_dim).astype(np.float32)
    #batch_size = 10
    n_hidden = tf.placeholder(tf.float32, name="n_hidden")
    n_hidden_value = np.sum(30).astype(np.float32)
    g_num_layers = tf.placeholder(tf.float32, name="g_num_layers")
    g_num_layers_value = np.sum(10).astype(np.float32)
    #d_num_layers = 10
    n_sample_z = 1
    seq_size_z = 15
    Z_dim = 10  # 10 dimension of a random input vector

    start_date = str(date_year)+'-'+str(date_month)+'-'+str(date_day)

    User_No = int(file_name[-8:-5])
    df = pd.read_excel(file_name)
    df = check_date_data_exist(df, start_date)
    date_first_str = str(df['Date'].tolist()[0])
    print("date_first_str = ", date_first_str)

    Weather_file_name = 'Weather_info/' + str(date_year) + '-' + str(date_month) + '.xlsx'
    Date_data = readExcel(Weather_file_name, 'ObsTime')
    Date_data_arr = np.vstack((Date_data))
    Date_data_list = []
    start_date_day = int(date_day) - 1
    for date_index in range(start_date_day, len(Date_data_arr)):
        if len(str(Date_data_arr[date_index][0])) < 2:
            Date_data_list.append(str(date_year) + '-' + str(date_month) + '-0' + str(Date_data_arr[date_index][0]))
        else:
            Date_data_list.append(str(date_year) + '-' + str(date_month) + '-' + str(Date_data_arr[date_index][0]))

    window_size = min(window_size, len(Date_data_list))
    Date_Generated = np.vstack(([Date_data_list[i] for i in range(window_size)]))
    print("Date_Generated = ", Date_Generated)
    #print("Date_Generated[0][0] = ", Date_Generated[0][0])

    #date_today = datetime.datetime.today()
    #date_today_str = date_today.strftime("%Y/%m/%d")

    d1 = datetime.datetime(int(Date_Generated[0][0][:4]), int(Date_Generated[0][0][5:7]), int(Date_Generated[0][0][8:10]))  # 第一个日期
    d2 = datetime.datetime(int(date_first_str[:4]), int(date_first_str[5:7]), int(date_first_str[8:10]))  # 第二个日期
    interval = d1 - d2  # 两日期差距
    new_date_diff = interval.days
    print("new_date_diff = ", new_date_diff)

    Date_Diff_Generated_1 = np.vstack(([new_date_diff+i for i in range(window_size)])) # for showing the right date_diff value
    Date_Diff_Generated = np.vstack(([i for i in range(window_size)])) # for calculation (date_diff should be in 0~15 for better prediction)
    Temperature_data = (readExcel(Weather_file_name, 'Temperature')).astype(np.float64)
    Temperature_Generated = np.vstack(([Temperature_data[start_date_day + i] for i in range(window_size)]))
    Humidity_data = (readExcel(Weather_file_name, 'RH')).astype(np.float64)
    Humidity_Generated = np.vstack(([Humidity_data[start_date_day + i] for i in range(window_size)]))


    #future_weather_info = get_future_weather(logitude, latitude)
    #Date_Diff_Generated = np.vstack(([new_date_diff+i for i in range(window_size)]))
    #Temperature_Generated = np.vstack(([future_weather_info[i][0] for i in range(window_size)]))
    #Humidity_Generated = np.vstack(([future_weather_info[i][1] for i in range(window_size)]))
    SkincareRatio_Generated = np.vstack(([random.randint(1, 7) for i in range(window_size)]))
    Age_Generated = np.vstack(([df['Age'].tolist()[-1] for i in range(window_size)]))

    #print("Date_Diff_Generated = ", Date_Diff_Generated)
    #print("Age_Generated = ", Age_Generated)

    Sample_y_data = np.stack(
        (Date_Diff_Generated, Temperature_Generated, Humidity_Generated, SkincareRatio_Generated, Age_Generated),
        axis=1)
    Sample_y_data = [Sample_y_data[i].reshape(1, -1) for i in range(len(Sample_y_data))]
    Sample_y_data = np.hstack((Sample_y_data)) / 100
    Sample_y_data = Sample_y_data.reshape([-1, 15, 5])
    print("Sample_y_data = ", Sample_y_data)
    # input("======")

    #Z_sample = sample_Z(n_sample_z, seq_size_z, Z_dim)
    y_sample = Sample_y_data

    user_No_code_table_name = 'User_No_Code_Table/User_No_code_table.xlsx'
    Z_sample = [increase_existing_user_data(user_No_code_table_name, User_No)[0][1:]]
    Z_sample = np.asarray(Z_sample).reshape(n_sample_z, seq_size_z, Z_dim)
    print("Z_sample = ", Z_sample)
    samples = sess.run(G_sample, feed_dict={"Z:0": Z_sample, "y:0": y_sample, keep_prob: keep_prob_value,
                                            seq_size: seq_size_value, n_hidden: n_hidden_value,
                                            g_num_layers: g_num_layers_value, input_size: input_size_value})
    samples = np.asarray(samples).reshape(window_size, -1)
    y_sample = np.asarray(y_sample).reshape(window_size, -1)  # shape (?, 4) for 4 fators in y
    #for i in range(len(y_sample)):
    #    date_shift = date_today_datefrom + datetime.timedelta(days=int(Date_Diff_Generated.flatten()[i]))
    #    date_shift_str = str(
    #        str(date_shift)[:4] + str('-') + str(date_shift)[5:7] + str('-') + str(date_shift)[8:10])
    #    date.append(date_shift_str)
    #date = np.vstack((date))
    user_data_generated = np.hstack((y_sample, samples))
    user_data_generated = user_data_generated * 100
    date_user_data_generated_header = ['Date', 'Date_Diff', 'Temperature', 'Humidity', 'Skincare_Ratio', 'Age',
                                       'Avg3_Hydration', 'Avg3_Skinhealth']
    data_generated = {
        date_user_data_generated_header[0]: Date_Generated.flatten(),
        date_user_data_generated_header[1]: Date_Diff_Generated_1.flatten(),
        date_user_data_generated_header[2]: user_data_generated[:, 1],
        date_user_data_generated_header[3]: user_data_generated[:, 2],
        date_user_data_generated_header[4]: user_data_generated[:, 3],
        date_user_data_generated_header[5]: user_data_generated[:, 4],
        date_user_data_generated_header[6]: user_data_generated[:, 5],
        date_user_data_generated_header[7]: user_data_generated[:, 6],
    }
    date_user_data_generated_df = pd.DataFrame(data_generated)
    date_user_data_generated_df_add = df.append(date_user_data_generated_df)
    date_user_data_generated_df_add.to_excel(file_name, index=None)
#
    #    user_code_list.append(Z_sample)
    #    user_No_list.append(int(file_name[-3:]))
#
    #    User_No = str(int(file_name[-3:]) + 1)
    #    length_zeros = 3 - len(User_No)
#
    #    for _ in range(length_zeros):
    #        User_No = '0' + User_No
    #    file_name = file_name[:-3] + User_No
#
#
    #user_code_table_header = ['User_No', 'User_code']
    #code_table_generated = {
    #    user_code_table_header[0]: user_No_list,
    #    user_code_table_header[1]: user_code_list,
    #}
    #table_file_name = 'User_No_code_table'
    #date_user_data_generated_df = pd.DataFrame(code_table_generated)
    #date_user_data_generated_df.to_excel('Generated_User_Data/' + table_file_name + '.xlsx', index=None)

    return sess, G_sample

def generate_FID_test_user_data(sess, G_sample, start_file_name_No, n_sample, window_size, Age, weather_file_name):

    Sample_file_name = 'Weather_info/'+str(weather_file_name)+'.xlsx'
    #Date_data = readExcel(Sample_file_name, 'ObsTime')
    #Date_data_arr = np.vstack((Date_data))
    #Date_data_list = []
    #for date_index in range(len(Date_data_arr)):
    #    if len(str(Date_data_arr[date_index][0])) < 2:
    #        Date_data_list.append(str(date_year)+'-'+str(date_month)+'-0'+str(Date_data_arr[date_index][0]))
    #    else:
    #        Date_data_list.append(str(date_year)+'-'+str(date_month)+'-'+str(Date_data_arr[date_index][0]))

    #Date_Generated = np.vstack(([Date_data_list[i] for i in range(window_size)]))
    #print("Date_Generated = ", Date_Generated)
    Z_dim = 10
    Date_data = (readExcel(Sample_file_name, 'Date')).astype(str)
    Date_Generated = np.vstack(([Date_data[i] for i in range(window_size)]))
    Date_Diff_data = (readExcel(Sample_file_name, 'Date_Diff')).astype(np.float64)
    Date_Diff_Generated = np.vstack(([Date_Diff_data[i] for i in range(window_size)]))
    Temperature_data = (readExcel(Sample_file_name, 'Temperature')).astype(np.float64)
    Temperature_Generated = np.vstack(([Temperature_data[i] for i in range(window_size)]))
    Humidity_data = (readExcel(Sample_file_name, 'RH')).astype(np.float64)
    Humidity_Generated = np.vstack(([Humidity_data[i] for i in range(window_size)]))
    Age_data = (readExcel(Sample_file_name, 'Age')).astype(np.float64)
    Age_Generated = np.vstack(([Age_data[i] for i in range(window_size)]))
    SkincareRatio_data = (readExcel(Sample_file_name, 'Skincare_Ratio')).astype(np.float64)
    SkincareRatio_Generated = np.vstack(([SkincareRatio_data[i]*10 for i in range(window_size)]))

    Sample_y_data = np.hstack(
        (Date_Diff_Generated, Temperature_Generated, Humidity_Generated, SkincareRatio_Generated, Age_Generated))

    Sample_y_data = [Sample_y_data[i].reshape(1, -1) for i in range(len(Sample_y_data))]
    Sample_y_data = np.hstack((Sample_y_data)) / 100
    print("Sample_y_data = ", Sample_y_data)


    file_name = 'GFID_User_' + start_file_name_No
    user_code_list = []
    user_No_list = []
    for i in range(n_sample):
        Z_sample = sample_Z(1, Z_dim)
        y_sample = list(Sample_y_data)
        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})
        samples = samples.reshape(window_size, -1)
        y_sample = np.asarray(y_sample).reshape(window_size, -1)  # shape (?, 4) for 4 fators in y
        #for i in range(len(y_sample)):
        #    date_shift = date_today_datefrom + datetime.timedelta(days=int(Date_Diff_Generated.flatten()[i]))
        #    date_shift_str = str(
        #        str(date_shift)[:4] + str('-') + str(date_shift)[5:7] + str('-') + str(date_shift)[8:10])
        #    date.append(date_shift_str)
        date = np.vstack((Date_Generated))
        user_data_generated = np.hstack((y_sample, samples))
        user_data_generated = user_data_generated * 100
        date_user_data_generated_header = ['Date', 'Day_Diff', 'Temperature', 'Humidity', 'Skincare_Ratio', 'Age',
                                           'Avg3_Hydration', 'Avg3_Skinhealth']

        data_generated = {
            date_user_data_generated_header[0]: date.flatten(),
            date_user_data_generated_header[1]: user_data_generated[:, 0],
            date_user_data_generated_header[2]: user_data_generated[:, 1],
            date_user_data_generated_header[3]: user_data_generated[:, 2],
            date_user_data_generated_header[4]: user_data_generated[:, 3]/10,
            date_user_data_generated_header[5]: user_data_generated[:, 4],
            date_user_data_generated_header[6]: user_data_generated[:, 5],
            date_user_data_generated_header[7]: user_data_generated[:, 6],
        }

        date_user_data_generated_df = pd.DataFrame(data_generated)
        date_user_data_generated_df.to_excel('Generated_User_Data/FID_Test_Data/' + file_name + '.xlsx', index=None)

        User_No = str(int(file_name[-3:]) + 1)
        length_zeros = 3 - len(User_No)

        for _ in range(length_zeros):
            User_No = '0' + User_No
        file_name = file_name[:-3] + User_No

def check_table_exist(file_name):
    if os.path.exists(file_name):
        print("111")
        return True
    else:
        print("222")
        return False

def increase_existing_user_data(file_name, User_No):
    table_exist = check_table_exist(file_name)
    if table_exist:
        df = pd.read_excel(file_name)
        user_code = df.loc[df['User_No'] == int(User_No)].values
        #print("user_code) = ", (user_code))
        return user_code
    else:
        print("Cannot find the file!")


def load_trained_CGAN_model(window_size, model_name):
    seq_size = 15  # for 15 days
    X_dim = 2  # hydration and oxygen
    y_dim = 5  # day_diff, temperature, humidity, skincareRatio, age
    Z_dim = 10  # 10 dimension of a random input vector
    input_size = X_dim
    batch_size = 10
    # For 180000 model
    #n_hidden = 20
    #g_num_layers = 5
    #d_num_layers = 5
    n_hidden = 30
    g_num_layers = 10
    d_num_layers = 10
    X = tf.placeholder(tf.float32, shape=[None, seq_size, X_dim], name="X")
    y = tf.placeholder(tf.float32, shape=[None, seq_size, y_dim], name='y')
    Z = tf.placeholder(tf.float32, shape=[None, seq_size, Z_dim], name="Z")
    keep_prob = np.sum(0.7).astype(np.float32)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    G_sample, g_params = generator(Z, y, keep_prob, seq_size, n_hidden, g_num_layers, input_size)
    y_data, y_generated, d_params = discriminator(X, G_sample, keep_prob, seq_size, n_hidden, d_num_layers, batch_size)
    D_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    # D_loss = tf.log(y_data) + tf.log(1 - y_generated)
    # D_loss = (tf.log(abs(y_data - y_generated)))
    G_loss = - tf.log(y_generated)
    # G_loss = tf.log(y_generated)
    # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    # D_loss = D_loss_real + D_loss_fake
    # G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    optimizer_g = tf.train.AdamOptimizer(0.0001)
    optimizer_d = tf.train.AdamOptimizer(0.00001)
    # optimizer_d = tf.train.AdamOptimizer(0.000001)
    D_solver = optimizer_d.minimize(D_loss, var_list=d_params)
    G_solver = optimizer_g.minimize(G_loss, var_list=g_params)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #'''restore model'''
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)  # 模型的保存器
    saver.restore(sess, 'Trained_CGAN/'+str(model_name))
    return sess, G_sample

if __name__ == '__main__':
    #Weather_file_name = '/Users/changtacheng/Desktop/Vescir_Edward/ICI_Project/CGAN_Model/VesCir_Skin_Data_Augmentation_System/Weather_Info/2019-01.xlsx'
    #Date_data = readExcel(Weather_file_name, 'ObsTime')
    #print("Date_data = ", Date_data)
    window_size = 15
    sess, G_sample = load_trained_CGAN_model(window_size)
    #sess = tf.Session()
    '''restore model'''
    #saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)  # 模型的保存器
    saver = tf.train.import_meta_graph('Trained_CGAN/model-499000.meta')
    saver.restore(sess, 'Trained_CGAN/model-499000')
    #sess.run(tf.global_variables_initializer())

    start_file_name_No = '001'
    n_sample = 100
    Age = 22
    # For Taiwan
    #logitude = 121
    #latitude = 25
    date_year = '2020'
    date_month = '01'
    date_day = '01'
    generate_new_user_data(sess, G_sample, start_file_name_No, n_sample, window_size, Age, date_year, date_month, date_day)
#
    #weather_file_name = 'Weather_Data_FID_Test'
    ##generate_FID_test_user_data(sess, G_sample, start_file_name_No, n_sample, window_size, Age, weather_file_name)
#
    #date_day = '16'
    #generate_existing_user_data(sess, G_sample,'Generated_User_Data/GUser_001.xlsx', window_size, date_year, date_month, date_day)

