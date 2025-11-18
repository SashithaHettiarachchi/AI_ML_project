%%%%main.m%%%%

clc; clear; close all;

% feature extraction
feature_extraction;

%train model only use day1 data(fd)
model_training;

%optimization
model_optimization;   

%model evaluation day2(md)
model_evaluation;


