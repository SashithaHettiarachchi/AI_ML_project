% model_training%%


clearvars; clc;
load(fullfile('Features','all_features.mat'),'all_feats','all_labels','all_sessions');

X = all_feats; y = all_labels; sess = all_sessions;
trainIdx = sess==1;     % Day1
Xtrain = X(trainIdx,:); ytrain = y(trainIdx);

% standardize
mu=mean(Xtrain,1); sigma=std(Xtrain,0,1)+eps;
Xtrain=(Xtrain-mu)./sigma;

[classes,~,yidx]=unique(ytrain);
T=full(ind2vec(yidx')); 

net=patternnet([64 32],'trainscg');
net.performFcn='crossentropy';
net.trainParam.epochs=150;
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.2;
net.divideParam.testRatio=0;
[net,tr]=train(net,Xtrain',T);

save(fullfile('Models','trained_net.mat'),'net','mu','sigma','classes');
