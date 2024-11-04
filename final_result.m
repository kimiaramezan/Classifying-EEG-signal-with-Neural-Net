clear all; clc; close all;

load("test_predict_labels.mat")
load("test_predict_labels2.mat")
test_predict_labels_final = [test_predict_labels ; test_predict_labels2];
save("test_predict_labels_final.mat","test_predict_labels_final")

find(abs(mean(test_predict_labels_final))==1)
