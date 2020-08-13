# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'VesCir_Skin_Data_Augmentation_System_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QBrush, QColor
import os
from CRNNCGAN_Loadmodel_test_UseSkincare import load_trained_CGAN_model, generate_new_user_data, generate_existing_user_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(918, 816)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(120, 0, 591, 51))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 50, 301, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(120, 340, 281, 16))
        font = QtGui.QFont()
        font.setFamily("AppleGothic")
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(400, 380, 101, 21))
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(400, 340, 111, 41))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(120, 360, 111, 16))
        self.label_5.setObjectName("label_5")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(120, 380, 111, 21))
        self.lineEdit_2.setText("")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(250, 360, 131, 16))
        self.label_6.setObjectName("label_6")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(250, 380, 131, 21))
        self.lineEdit_3.setText("")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(490, 390, 101, 41))
        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_7.setAutoFillBackground(False)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(490, 430, 101, 21))
        self.lineEdit_4.setText("")
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(120, 410, 111, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(240, 410, 121, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(370, 410, 101, 16))
        self.label_10.setObjectName("label_10")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(120, 430, 101, 21))
        self.lineEdit_5.setText("")
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(370, 430, 101, 21))
        self.lineEdit_6.setText("")
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_7.setGeometry(QtCore.QRect(240, 430, 111, 21))
        self.lineEdit_7.setText("")
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(600, 400, 113, 61))
        self.pushButton.setObjectName("pushButton")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(120, 490, 381, 16))
        font = QtGui.QFont()
        font.setFamily("AppleGothic")
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(120, 560, 111, 16))
        self.label_12.setObjectName("label_12")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(120, 510, 151, 16))
        self.label_14.setObjectName("label_14")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_8.setGeometry(QtCore.QRect(370, 580, 101, 21))
        self.lineEdit_8.setText("")
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_9.setGeometry(QtCore.QRect(370, 530, 101, 21))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(552, 550, 161, 61))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(240, 560, 121, 16))
        self.label_16.setObjectName("label_16")
        self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_10.setGeometry(QtCore.QRect(240, 580, 111, 21))
        self.lineEdit_10.setText("")
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_12 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_12.setGeometry(QtCore.QRect(120, 580, 101, 21))
        self.lineEdit_12.setText("")
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(370, 560, 101, 16))
        self.label_18.setObjectName("label_18")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(120, 470, 471, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(120, 610, 431, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(120, 140, 121, 16))
        font = QtGui.QFont()
        font.setFamily("AppleGothic")
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(120, 210, 471, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(320, 160, 113, 51))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(500, 130, 201, 41))
        self.label_17.setText("")
        self.label_17.setObjectName("label_17")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(530, 330, 361, 61))
        self.label_19.setText("")
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(520, 480, 391, 61))
        self.label_20.setText("")
        self.label_20.setObjectName("label_20")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(600, 170, 113, 51))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(260, 520, 81, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(120, 160, 191, 41))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(120, 530, 141, 21))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(120, 260, 191, 41))
        self.textEdit_3.setObjectName("textEdit_3")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(120, 240, 121, 16))
        font = QtGui.QFont()
        font.setFamily("AppleGothic")
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(120, 310, 471, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(500, 230, 201, 41))
        self.label_22.setText("")
        self.label_22.setObjectName("label_22")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(600, 270, 113, 51))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(320, 260, 113, 51))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(370, 500, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_23.setFont(font)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 918, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #####
        self.sess = None
        self.G_sample = None
        self.saver = None
        self.model_name = str()
        self.ExistUser_File = str()
        self.TrainedModel_File = str()
        self.WeatherInfo_File = str()
        self.window_size = 15  # Predicting days
        self.pushButton_3.clicked.connect(self.browseTrainedModel)
        self.pushButton_4.clicked.connect(self.LoadTrainedModel)
        self.pushButton.clicked.connect(self.genNewUserData)
        self.pushButton_5.clicked.connect(self.browseExistUser)
        self.pushButton_2.clicked.connect(self.addNewDataToExistUser)
        self.pushButton_7.clicked.connect(self.browseWeatherInfo)
        self.pushButton_6.clicked.connect(self.LoadWeatherInfo)

    def browseTrainedModel(self):
        self.textEdit.clear()  # initial item list in listWidget
        self.TrainedModel_File = str()
        cwd = os.getcwd()  # get current parent directory path
        # cwd = cwd+'/P2_Use_Skincare/Use_Skincare/'
        print("cwd = ", cwd)
        files, ok1 = QFileDialog.getOpenFileNames(None, "Choose file", cwd)
        if len(files) == 0:
            print("\nCancel")
            return
        print("\nChoose Files:")
        for file in files:
            print(file)
        print(files, ok1)
        self.TrainedModel_File = files[0].split(".")[0].split("/")[-1]
        self.textEdit.setText(self.TrainedModel_File)
        self.textEdit.repaint()

    def LoadTrainedModel(self):
        #self.sess = None
        #self.G_sample = None
        self.label_17.clear()
        self.label_17.repaint()
        #self.sess, self.G_sample = load_trained_CGAN_model(self.window_size)
        self.model_name = self.textEdit.toPlainText()
        if len(self.model_name) != 0:
            #self.saver = tf.train.import_meta_graph('Trained_CGAN/' + model_name + '.meta')
            #self.saver.restore(self.sess, 'Trained_CGAN/' + model_name)
            self.label_17.setText("Successfully load model!!")
            self.label_17.repaint()
        else:
            self.label_17.setText("Check 'Load Trained Model' step,\nmake sure at least one trained model is selected.")
            self.label_17.repaint()

    def browseWeatherInfo(self):
        self.textEdit_3.clear()  # initial item list in listWidget
        cwd = os.getcwd()  # get current parent directory path
        # cwd = cwd+'/P2_Use_Skincare/Use_Skincare/'
        print("cwd = ", cwd)
        files, ok1 = QFileDialog.getOpenFileNames(None, "Choose file", cwd)
        if len(files) == 0:
            print("\nCancel")
            return
        print("\nChoose Files:")
        for file in files:
            print(file)
        print(files, ok1)
        self.WeatherInfo_File = files[0].split(".")[0].split("/")[-1]
        self.textEdit_3.setText(self.WeatherInfo_File)
        self.textEdit_3.repaint()

    def LoadWeatherInfo(self):
        self.label_22.clear()
        self.label_22.repaint()
        self.lineEdit_5.clear()
        self.lineEdit_12.clear()
        self.lineEdit_7.clear()
        self.lineEdit_10.clear()
        self.lineEdit.clear()
        self.lineEdit_9.clear()
        self.lineEdit_6.clear()
        self.lineEdit_8.clear()
        if len(self.WeatherInfo_File) != 0:
            self.lineEdit_5.setText(self.WeatherInfo_File[:4])
            self.lineEdit_5.repaint()
            self.lineEdit_12.setText(self.WeatherInfo_File[:4])
            self.lineEdit_12.repaint()
            self.lineEdit_7.setText(self.WeatherInfo_File[-2:])
            self.lineEdit_7.repaint()
            self.lineEdit_10.setText(self.WeatherInfo_File[-2:])
            self.lineEdit_10.repaint()
            self.lineEdit.setText(str(self.window_size))
            self.lineEdit.repaint()
            self.lineEdit_9.setText(str(self.window_size))
            self.lineEdit_9.repaint()
            self.lineEdit_6.setText('01')
            self.lineEdit_6.repaint()
            self.lineEdit_8.setText('16')
            self.lineEdit_8.repaint()

            self.label_22.setText("Successfully load weather info.!!")
            self.label_22.repaint()
        else:
            self.label_22.setText("Check 'Load Weather Info.' step,\nmake sure at least one Weather Info file is selected.")
            self.label_22.repaint()

    def genNewUserData(self):
        self.label_19.clear()
        self.label_19.repaint()
        cwd = os.getcwd()  # get current parent directory path
        weather_cwd = cwd + '/Weather_Info/'
        #if self.sess == None or self.G_sample == None or self.saver == None:
        if len(self.model_name) == 0:
            self.label_19.setText("Check 'Load Trained Model' step,\nmake sure at least one trained model is selected.")
            self.label_19.repaint()
        elif len(self.WeatherInfo_File) == 0:
            self.label_19.setText(
                "Check 'Load Weather Info.' step,\nmake sure at least one Weather Info file is selected.")
            self.label_19.repaint()
        else:
            if len(self.lineEdit_2.text()) != 3 or len(self.lineEdit_3.text()) == 0 or len(
                    self.lineEdit.text()) == 0 or len(self.lineEdit_4.text()) == 0 \
                    or len(self.lineEdit_5.text()) != 4 or len(
                self.lineEdit_7.text()) != 2 or len(self.lineEdit_6.text()) != 2:
                self.label_19.setText(
                    "Make sure all items are\nin the right forms respectively.")
                self.label_19.repaint()
            else:
                print("weather_cwd + self.lineEdit_5.text() + '-' + self.lineEdit_7.text() + '.xlsx' = ",
                      weather_cwd + self.lineEdit_5.text() + '-' + self.lineEdit_7.text() + '.xlsx')
                if os.path.isfile(
                        weather_cwd + self.lineEdit_5.text() + '-' + self.lineEdit_7.text() + '.xlsx'):  # if the weather file exists
                    start_file_name_No = self.lineEdit_2.text()
                    n_sample = int(self.lineEdit_3.text())
                    window_size = int(self.lineEdit.text())
                    Age = int(self.lineEdit_4.text())
                    date_year = self.lineEdit_5.text()
                    date_month = self.lineEdit_7.text()
                    date_day = self.lineEdit_6.text()

                    self.sess, self.G_sample = generate_new_user_data(self.sess, self.G_sample, self.model_name, start_file_name_No, n_sample, window_size, Age,
                                           date_year,
                                           date_month,
                                           date_day)

                    self.label_19.setText("Successfully generate some user data!!")
                    self.label_19.repaint()
                else:
                    self.label_19.setText("There is no matched weather file!\nCheck the values of Year and Month!")
                    self.label_19.repaint()

    def browseExistUser(self):
        self.label_20.clear()
        self.label_20.repaint()
        cwd = os.getcwd()  # get current parent directory path
        # cwd = cwd+'/P2_Use_Skincare/Use_Skincare/'
        print("cwd = ", cwd)
        files, ok1 = QFileDialog.getOpenFileNames(None, "Choose file", cwd, r'Excel Files(*.xlsx)')
        if len(files) == 0:
            print("\nCancel")
            return
        print("\nChoose Files:")
        for file in files:
            print(file)
        print(files, ok1)
        self.ExistUser_File = files[0]
        self.textEdit_2.setText(self.ExistUser_File.split(".")[0].split("/")[-1])
        self.textEdit_2.repaint()

    def addNewDataToExistUser(self):
        self.label_20.clear()
        self.label_20.repaint()
        cwd = os.getcwd()  # get current parent directory path
        weather_cwd = cwd + '/Weather_Info/'
        if len(self.model_name) == 0:
            self.label_20.setText("Check 'Load Trained Model' step,\nmake sure at least one trained model is selected.")
            self.label_20.repaint()
        elif len(self.WeatherInfo_File) == 0:
            self.label_20.setText(
                "Check 'Load Weather Info.' step,\nmake sure at least one Weather Info file is selected.")
            self.label_20.repaint()
        elif len(self.ExistUser_File) == 0:
            self.label_20.setText(
                "Make sure at least one existing user file is selected.")
            self.label_20.repaint()
        else:
            if len(self.lineEdit_9.text()) == 0 or len(self.lineEdit_12.text()) != 4 or len(
                    self.lineEdit_10.text()) != 2 or len(self.lineEdit_8.text()) != 2:
                self.label_20.setText(
                    "Make sure all items are\nin the right forms respectively.")
                self.label_20.repaint()
            else:
                if os.path.isfile(
                        weather_cwd + self.lineEdit_12.text() + '-' + self.lineEdit_10.text() + '.xlsx'):  # if the weather file exists

                    window_size = int(self.lineEdit_9.text())
                    date_year = self.lineEdit_12.text()
                    date_month = self.lineEdit_10.text()
                    date_day = self.lineEdit_8.text()

                    self.sess, self.G_sample = generate_existing_user_data(self.sess, self.G_sample, self.model_name, self.ExistUser_File, window_size, date_year,
                                                date_month,
                                                date_day)

                    self.label_20.setText("Successfully generate some user data\nto existing user file!!")
                    self.label_20.repaint()
                else:
                    self.label_20.setText("There is no matched weather file!\nCheck the values of Year and Month!")
                    self.label_20.repaint()

    #####

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "VesCir Ltd. Skin Data Augmentation Algorithm Testing System"))
        self.label_2.setText(_translate("MainWindow", "Ver. 1.0 Developed by Edward Chang 20200323"))
        self.label_3.setText(_translate("MainWindow", "Input Condition Setting(For generating new file)"))
        self.label_4.setText(_translate("MainWindow", "Predicting Day(s)\n"
"(Max:15)"))
        self.label_5.setText(_translate("MainWindow", "First Gen. File No."))
        self.label_6.setText(_translate("MainWindow", "Numbers of Gen. File"))
        self.label_7.setText(_translate("MainWindow", "Avg. Age\n"
"in Gen. Files"))
        self.label_8.setText(_translate("MainWindow", "Start Date (Year)"))
        self.label_9.setText(_translate("MainWindow", "Start Date (Month)"))
        self.label_10.setText(_translate("MainWindow", "Start Date (Day)"))
        self.pushButton.setText(_translate("MainWindow", "Start\n"
"Generating\n"
"New Files"))
        self.label_11.setText(_translate("MainWindow", "Input Condition Setting(For generating new data to existing file)"))
        self.label_12.setText(_translate("MainWindow", "Start Date (Year)"))
        self.label_14.setText(_translate("MainWindow", "Choose one existing file"))
        self.pushButton_2.setText(_translate("MainWindow", "Start\n"
"Generating New Data\n"
"to existing Files"))
        self.label_16.setText(_translate("MainWindow", "Start Date (Month)"))
        self.label_18.setText(_translate("MainWindow", "Start Date (Day)"))
        self.label_15.setText(_translate("MainWindow", "Load Trained Model"))
        self.pushButton_3.setText(_translate("MainWindow", "Browse\n"
"Trained Model"))
        self.pushButton_4.setText(_translate("MainWindow", "Load\n"
"Trained Model"))
        self.pushButton_5.setText(_translate("MainWindow", "Browse"))
        self.label_21.setText(_translate("MainWindow", "Load Weather Info."))
        self.pushButton_6.setText(_translate("MainWindow", "Load\n"
"Weather Info."))
        self.pushButton_7.setText(_translate("MainWindow", "Browse\n"
"Weather File"))
        self.label_23.setText(_translate("MainWindow", "Predicting Day(s)\n"
"(Max:15)"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

