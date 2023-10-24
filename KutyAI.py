#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 23:09:01 2023

@author: kmark7
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
import tensorflow as tf
import numpy as np
import pathlib

model2 = tf.keras.models.load_model("KutyaJoModel")

def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3) #tensor formatum
    img = tf.image.resize(img, size = [img_shape, img_shape])
    img = tf.expand_dims(img, axis=0)
    img -= [103, 116, 123]
    img = img/255.
    return img

class FileDialogExample(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 200)
        self.setWindowTitle('KutyAI - a fajtafelismerő')

        self.button = QPushButton('Kép kiválasztása', self)
        self.button.setGeometry(300, 150, 200, 50)
        self.button.clicked.connect(self.showDialog)

    
        self.label = QLabel('', self)
        self.label.setGeometry(10, 10, 700, 30)
        
        self.label2 = QLabel('', self)
        self.label2.setGeometry(10, 50, 380, 30)
        
        self.label3 = QLabel('', self)
        self.label3.setGeometry(10, 70, 380, 30)
        
        self.label4 = QLabel('', self)
        self.label4.setGeometry(10, 90, 380, 30)
        
        
        self.label2p = QLabel('', self)
        self.label2p.setGeometry(400, 50, 380, 30)
        
        self.label3p = QLabel('', self)
        self.label3p.setGeometry(400, 70, 380, 30)
        
        self.label4p = QLabel('', self)
        self.label4p.setGeometry(400, 90, 380, 30)
        

    def showDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_path, _ = QFileDialog.getOpenFileName(self, 'Képfájl kiválasztása', '', 'Képfájlok (*.jpg *.jpeg *.png *.bmp *.gif);;Minden fájl (*)', options=options)

        if file_path:
            print('Kiválasztott képfájl elérési útja:', file_path)
            
            image = load_and_prep_image(file_path)
            pred = model2.predict(image)

            data_dir = pathlib.Path("dogs/train")
            class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
            #print(class_names) inkább ne

            x = pred[0][np.argmax(pred[0])]
            pred_class = class_names[np.argmax(pred[0])]
            pred_class_p = str((x*100).round(2))+" %"
            
            P2 = np.delete(pred[0],np.argmax(pred[0]))
            y = P2[np.argmax(P2)]
            second_pred_class = class_names[np.argmax(P2)]
            second_pred_class_p = str((y*100).round(2))+" %"
            
            P3 = np.delete(P2,np.argmax(P2))
            z = P3[np.argmax(P3)]
            thirdy_pred_class = class_names[np.argmax(P3)]
            thirdy_pred_class_p = str((z*100).round(2))+" %"
            
            #pred_class
            
            self.label.setText('A kiválasztott fájl: ' + file_path)
            self.label2.setText('TOP 1. kutyafajta neve: ' + pred_class)
            self.label3.setText('TOP 2. kutyafajta neve: ' + second_pred_class)
            self.label4.setText('TOP 3. kutyafajta neve: ' + thirdy_pred_class)
            
            self.label2p.setText(' és százalékos tartalma: ' + pred_class_p)
            self.label3p.setText(' és százalékos tartalma: ' + second_pred_class_p)
            self.label4p.setText(' és százalékos tartalma: ' + thirdy_pred_class_p)
            
def main():
    app = QApplication(sys.argv)
    ex = FileDialogExample()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
