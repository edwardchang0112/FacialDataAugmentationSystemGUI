import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import pandas as pd
import datetime
import csv
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

def generator(z, y, keep_prob, G_W1, G_W2, G_W3, G_W4, G_W5, G_W6, G_W7, G_W8, G_b1, G_b2, G_b3, G_b4, G_b5, G_b6, G_b7, G_b8):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(inputs, G_W1) + G_b1), keep_prob)
    G_h2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(G_h1, G_W2) + G_b2), keep_prob)
    G_h3 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(G_h2, G_W3) + G_b3), keep_prob)
    G_h4 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(G_h3, G_W4) + G_b4), keep_prob)
    G_h5 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(G_h4, G_W5) + G_b5), keep_prob)
    G_h6 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(G_h5, G_W6) + G_b6), keep_prob)
    G_h7 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(G_h6, G_W7) + G_b7), keep_prob)
    G_log_prob = tf.matmul(G_h7, G_W8) + G_b8
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

def discriminator(x, y, keep_prob, D_W1, D_W2, D_W3, D_W4, D_W5, D_W6, D_W7, D_W8, D_b1, D_b2, D_b3, D_b4, D_b5, D_b6, D_b7, D_b8):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(inputs, D_W1) + D_b1), keep_prob)
    D_h2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(D_h1, D_W2) + D_b2), keep_prob)
    D_h3 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(D_h2, D_W3) + D_b3), keep_prob)
    D_h4 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(D_h3, D_W4) + D_b4), keep_prob)
    D_h5 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(D_h4, D_W5) + D_b5), keep_prob)
    D_h6 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(D_h5, D_W6) + D_b6), keep_prob)
    D_h7 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(D_h6, D_W7) + D_b7), keep_prob)
    D_logit = tf.matmul(D_h7, D_W8) + D_b8
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

def sample_Z(m, n): #生成维度为[m, n]的随机噪声作为生成器G的输入
    return np.random.normal(0., 0.1, size=[m, n])

def xavier_init(size): #初始化参数时使用的xavier_init函数
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.) #初始化标准差
    return tf.random_normal(shape=size, stddev=xavier_stddev) #返回初始化的结果

def save(saver, sess, logdir, step): #保存模型的save函数
    model_name = 'model' #模型名前缀
    checkpoint_path = os.path.join(logdir, model_name) #保存路径
    saver.save(sess, checkpoint_path, global_step=step) #保存模型
    print('The checkpoint has been created.')

def generate_new_user_data(sess, G_sample, start_file_name_No, n_sample, window_size, Age, date_year, date_month, date_day):
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


    Z_dim = 10
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
        print("Sample_y_data = ", Sample_y_data)

        Z_sample = sample_Z(1, Z_dim)
        y_sample = Sample_y_data
        samples = sess.run(G_sample, feed_dict={"Z:0": Z_sample, "y:0": y_sample})
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
        date_user_data_generated_header = ['Date', 'Date_Diff', 'Temperature', 'Humidity', 'Skincare_Ratio', 'Age',
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
        date_user_data_generated_df.to_excel('Generated_User_Data/' + file_name + '.xlsx', index=None)

        #print("Z_sample = ", Z_sample)
        #print("Z_sample[0] = ", Z_sample[0])
        user_code_list.append(Z_sample[0])
        #print("user_code_list = ", user_code_list)
        #input("==========")
        user_No_list.append(int(file_name[-3:]))

        User_No = str(int(file_name[-3:]) + 1)
        length_zeros = 3 - len(User_No)

        for _ in range(length_zeros):
            User_No = '0' + User_No
        file_name = file_name[:-3] + User_No

    user_code_arr = np.vstack((user_code_list))

    create_user_No_code_table(user_No_list, user_code_arr)

def check_table_exists(table_file_name):
    if os.path.exists(table_file_name):
        print(table_file_name+" exists!")
        return True
    else:
        print(table_file_name+" not exists!")
        return False

def create_user_No_code_table(user_No_list, user_code_list):
    user_code_table_header = ['User_No', 'User_code_0', 'User_code_1', 'User_code_2', 'User_code_3', 'User_code_4',
                              'User_code_5', 'User_code_6', 'User_code_7', 'User_code_8', 'User_code_9']
    code_table_generated = {
        user_code_table_header[0]: user_No_list,
        user_code_table_header[1]: user_code_list[:, 0],
        user_code_table_header[2]: user_code_list[:, 1],
        user_code_table_header[3]: user_code_list[:, 2],
        user_code_table_header[4]: user_code_list[:, 3],
        user_code_table_header[5]: user_code_list[:, 4],
        user_code_table_header[6]: user_code_list[:, 5],
        user_code_table_header[7]: user_code_list[:, 6],
        user_code_table_header[8]: user_code_list[:, 7],
        user_code_table_header[9]: user_code_list[:, 8],
        user_code_table_header[10]: user_code_list[:, 9],
    }
    table_file_name = 'Generated_User_Data/User_No_code_table.xlsx'
    table_exists = check_table_exists(table_file_name)
    print("code_table_generated = ", code_table_generated)
    date_user_data_generated_df = pd.DataFrame(code_table_generated)
    print("date_user_data_generated_df = ", date_user_data_generated_df)
    if table_exists:
        df = pd.read_excel(table_file_name, index=None)
        #print("code_table_generated.values() = ", np.stack((code_table_generated.values())).T)
        print("np.stack((df.values)) = ", np.stack((df.values)))
        print("np.stack((code_table_generated.values())).T = ", np.stack((code_table_generated.values())).T)
        df_1 = np.vstack((np.stack((df.values)), np.stack((code_table_generated.values())).T))
        df_2 = pd.DataFrame(df_1)
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

def generate_existing_user_data(sess, G_sample, file_name, window_size, date_year, date_month, date_day):
    #start_file_name_No = '001'
    #n_sample = 10
    #Age = 35
    ## For Taiwan
    #logitude = 121
    #latitude = 25

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

    #start_date = date_today_str
    #date_today_datefrom = datetime.date(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:10]))
    #date = []
    #Z_sample = sample_Z(1, Z_dim)
    user_No_code_table_name = 'Generated_User_Data/User_No_code_table.xlsx'
    Z_sample = [increase_existing_user_data(user_No_code_table_name, User_No)[0][1:]]
    print("Z_sample = ", Z_sample)
    y_sample = list(Sample_y_data)
    samples = sess.run(G_sample, feed_dict={"Z:0": Z_sample, "y:0": y_sample})
    samples = samples.reshape(window_size, -1)
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


def load_trained_CGAN_model(window_size):
    #window_size = 15 # How many days of data can be generated.
    #mb_size = 10
    Z_dim = 10
    X_dim = 2 * window_size # 2 stands for hydration and oxygen
    y_dim = 5 * window_size # 5 stands for Date_Diff, Temperature, Humidity, SkincareRatio, Age

    D_h1_dim = 256
    D_h2_dim = 128
    D_h3_dim = 64
    D_h4_dim = 32
    D_h5_dim = 16
    D_h6_dim = 8
    D_h7_dim = 4

    G_h1_dim = 4
    G_h2_dim = 8
    G_h3_dim = 16
    G_h4_dim = 32
    G_h5_dim = 64
    G_h6_dim = 128
    G_h7_dim = 256
    keep_prob = 0.8

    ''' Discriminator Net model '''
    X = tf.placeholder(tf.float32, shape=[None, X_dim], name='X')
    y = tf.placeholder(tf.float32, shape=[None, y_dim], name='y')

    D_W1 = tf.Variable(xavier_init([X_dim + y_dim, D_h1_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[D_h1_dim]))

    D_W2 = tf.Variable(xavier_init([D_h1_dim, D_h2_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[D_h2_dim]))

    D_W3 = tf.Variable(xavier_init([D_h2_dim, D_h3_dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[D_h3_dim]))

    D_W4 = tf.Variable(xavier_init([D_h3_dim, D_h4_dim]))
    D_b4 = tf.Variable(tf.zeros(shape=[D_h4_dim]))

    D_W5 = tf.Variable(xavier_init([D_h4_dim, D_h5_dim]))
    D_b5 = tf.Variable(tf.zeros(shape=[D_h5_dim]))

    D_W6 = tf.Variable(xavier_init([D_h5_dim, D_h6_dim]))
    D_b6 = tf.Variable(tf.zeros(shape=[D_h6_dim]))

    D_W7 = tf.Variable(xavier_init([D_h6_dim, D_h7_dim]))
    D_b7 = tf.Variable(tf.zeros(shape=[D_h7_dim]))

    D_W8 = tf.Variable(xavier_init([D_h7_dim, 1]))
    D_b8 = tf.Variable(tf.zeros(shape=[1]))

    # D_W8 = tf.Variable(xavier_init([D_h7_dim, 1]))
    # D_b8 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_W3, D_W4, D_W5, D_W6, D_W7, D_W8, D_b1, D_b2, D_b3, D_b4, D_b5, D_b6, D_b7, D_b8]

    ''' Generator Net model '''
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim], name='Z')

    G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, G_h1_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[G_h1_dim]))

    G_W2 = tf.Variable(xavier_init([G_h1_dim, G_h2_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[G_h2_dim]))

    G_W3 = tf.Variable(xavier_init([G_h2_dim, G_h3_dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[G_h3_dim]))

    G_W4 = tf.Variable(xavier_init([G_h3_dim, G_h4_dim]))
    G_b4 = tf.Variable(tf.zeros(shape=[G_h4_dim]))

    G_W5 = tf.Variable(xavier_init([G_h4_dim, G_h5_dim]))
    G_b5 = tf.Variable(tf.zeros(shape=[G_h5_dim]))

    G_W6 = tf.Variable(xavier_init([G_h5_dim, G_h6_dim]))
    G_b6 = tf.Variable(tf.zeros(shape=[G_h6_dim]))

    G_W7 = tf.Variable(xavier_init([G_h6_dim, G_h7_dim]))
    G_b7 = tf.Variable(tf.zeros(shape=[G_h7_dim]))

    G_W8 = tf.Variable(xavier_init([G_h7_dim, X_dim]))
    G_b8 = tf.Variable(tf.zeros(shape=[X_dim]))

    theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_W6, G_W7, G_W8, G_b1, G_b2, G_b3, G_b4, G_b5, G_b6, G_b7, G_b8]

    G_sample = generator(Z, y, keep_prob, G_W1, G_W2, G_W3, G_W4, G_W5, G_W6, G_W7, G_W8, G_b1, G_b2, G_b3, G_b4, G_b5, G_b6, G_b7, G_b8)
    D_real, D_logit_real = discriminator(X, y, keep_prob, D_W1, D_W2, D_W3, D_W4, D_W5, D_W6, D_W7, D_W8, D_b1, D_b2, D_b3, D_b4, D_b5, D_b6, D_b7, D_b8)
    D_fake, D_logit_fake = discriminator(G_sample, y, keep_prob, D_W1, D_W2, D_W3, D_W4, D_W5, D_W6, D_W7, D_W8, D_b1, D_b2, D_b3, D_b4, D_b5, D_b6, D_b7, D_b8)

    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    D_solver = tf.train.AdamOptimizer(0.001).minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss, var_list=theta_G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #'''restore model'''
    #saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)  # 模型的保存器
    #saver.restore(sess, 'Trained_CGAN/model-499000')
    return sess, G_sample

if __name__ == '__main__':

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
    Age = 35
    # For Taiwan
    #logitude = 121
    #latitude = 25
    date_year = '2020'
    date_month = '01'
    date_day = '01'
    generate_new_user_data(sess, G_sample, start_file_name_No, n_sample, window_size, Age, date_year, date_month, date_day)

    weather_file_name = 'Weather_Data_FID_Test'
    #generate_FID_test_user_data(sess, G_sample, start_file_name_No, n_sample, window_size, Age, weather_file_name)

    date_day = '16'
    generate_existing_user_data(sess, G_sample,'Generated_User_Data/GUser_001.xlsx', window_size, date_year, date_month, date_day)

