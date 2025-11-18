
% model_optimization%%

clc; clear; close all;

% Load features
load(fullfile('Features','all_features.mat'),'all_feats','all_labels','all_sessions');

X = all_feats; 
y = all_labels; 
sess = all_sessions;

% Use only Day-1 as training data
trainIdx = (sess == 1);
Xtrain = X(trainIdx,:);
ytrain = y(trainIdx);

% Standardization
mu = mean(Xtrain,1);
sigma = std(Xtrain,0,1) + eps;
Xtrain = (Xtrain - mu) ./ sigma;

% Classes → one-hot
[classes,~,yidx] = unique(ytrain);
T = full(ind2vec(yidx'));


% Objective Functio

optimFun = @(params) nn_objective(params, Xtrain, T);


vars = [
    optimizableVariable('H1',[16 128],'Type','integer')
    optimizableVariable('H2',[16 128],'Type','integer')
    optimizableVariable('LR',[1e-4 5e-2],'Transform','log')
    optimizableVariable('Epochs',[80 250],'Type','integer')
];


% Run Optimization

results = bayesopt(optimFun, vars, ...
        'MaxObjectiveEvaluations', 25, ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'Verbose',1, ...
        'PlotFcn', {@plotObjectiveModel,@plotMinObjective});

bestParams = results.XAtMinObjective;

disp(bestParams);


% Train Final Optimized Network

fprintf("\nTraining final optimized network...\n");

net = patternnet([bestParams.H1 bestParams.H2], 'trainscg');
net.performFcn = 'crossentropy';
net.trainParam.lr = bestParams.LR;
net.trainParam.epochs = bestParams.Epochs;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;

[net, ~] = train(net, Xtrain', T);

% Save optimized model%%
if ~exist('Models','dir'), mkdir('Models'); end
save(fullfile('Models','trained_net_optimized.mat'), ...
     'net','mu','sigma','classes','bestParams');




%%%Objective function for optimization%%

function err = nn_objective(params, Xtrain, T)

    % Build candidate network
    net = patternnet([params.H1 params.H2], 'trainscg');
    net.performFcn = 'crossentropy';
    net.trainParam.lr = params.LR;
    net.trainParam.epochs = params.Epochs;

    % No test split, only train for speed
    net.divideParam.trainRatio = 0.75;
    net.divideParam.valRatio   = 0.25;
    net.divideParam.testRatio  = 0;

    % Train
    [net, tr] = train(net, Xtrain', T, 'useParallel','no','showResources','no');

    % Return validation error → minimized
    err = tr.best_vperf;
end
