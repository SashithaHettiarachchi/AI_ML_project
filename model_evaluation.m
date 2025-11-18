
%%%%model_evaluatin_part%%%%
clc; clear;


% Load data
load(fullfile('Features','all_features.mat'), 'all_feats', 'all_labels', 'all_sessions');
load(fullfile('Models','trained_net.mat'), 'net', 'mu', 'sigma', 'classes');
load(fullfile('Models','trained_net_optimized.mat'), 'net', 'mu', 'sigma', 'classes');



X = all_feats; 
y = all_labels; 
sess = all_sessions;

testIdx = (sess == 2);   %  test only Day MD
fprintf('Total samples: %d | Day-2 samples: %d\n', size(X,1), sum(testIdx));

if sum(testIdx) == 0
    error(' Nosamples found.');
end

Xtest = (X(testIdx,:) - mu) ./ sigma;
ytest = y(testIdx);

% NN predict
Yprob = net(Xtest');
[~, predIdx] = max(Yprob, [], 1);
[~, ~, yidxTrue] = unique(ytest);

confmat = confusionmat(yidxTrue, predIdx');

if isempty(confmat)
    error('empty');
end

figure;
confusionchart(confmat, cellstr(string(classes)));
title('Confusion Matrix – Day2 (MD) Evaluation');
xlabel('Predicted Class'); ylabel('True Class');

% accuracy
acc = sum(diag(confmat)) / sum(confmat(:));
fprintf('Accuracy on Day-2 data = %.3f\n', acc);

% FAR / FRR / EER 
nC = size(confmat,1);
FAR = zeros(nC,1); FRR = zeros(nC,1); EER = zeros(nC,1);

for i = 1:nC
    TP = confmat(i,i);
    FN = sum(confmat(i,:)) - TP;
    FP = sum(confmat(:,i)) - TP;
    TN = sum(confmat(:)) - (TP + FN + FP);

    FAR(i) = FP / (FP + TN + eps);
    FRR(i) = FN / (TP + FN + eps);

    % EER calculation using ROC curve
    scores = Yprob(i,:)';
    labels = (yidxTrue == i);
    [fpr, tpr] = perfcurve(labels, scores, 1);
    fnr = 1 - tpr;
    [~, idx] = min(abs(fpr - fnr));
    EER(i) = (fpr(idx) + fnr(idx)) / 2;
end

fprintf('Average FAR = %.3f | Average FRR = %.3f | Average EER = %.3f\n', ...
        mean(FAR), mean(FRR), mean(EER));


timestamp = datestr(now,'yyyymmdd_HHMMSS');
PLOT_DIR = fullfile('Models','Plots');
if ~exist(PLOT_DIR,'dir'), mkdir(PLOT_DIR); end


fig1 = figure('Name','All Evaluation Plots','Position',[100 100 1200 800]);

subplot(2,2,1);   % Confusion Matrix
confusionchart(confmat, cellstr(string(classes)));
title('Confusion Matrix – Day2 (MD Evaluation)');


subplot(2,2,2); % ROC + EER
hold on; grid on;
colors = lines(nC);
legend_labels = cell(nC,1);

for i = 1:nC
    scores = Yprob(i,:)';
    labels = (yidxTrue == i);

    [fpr, tpr] = perfcurve(labels, scores, 1);
    fnr = 1 - tpr;
    [~, idx] = min(abs(fpr - fnr));
    eer_point = fpr(idx);

    plot(fpr, tpr, 'Color', colors(i,:), 'LineWidth', 1.5);
    plot(fpr(idx), tpr(idx), 'o','MarkerSize',6,'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor','k');

    legend_labels{i} = sprintf('User %d (EER=%.3f)', classes(i), eer_point);
end
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves with EER Points');
legend(legend_labels,'Location','southoutside');
hold off;

% PCA Scatter Plot
subplot(2,2,3);
[coeff, score] = pca(Xtest);
gscatter(score(:,1), score(:,2), ytest);
xlabel('PC1'); ylabel('PC2');
title('PCA Scatter Plot (Feature Space)');
grid on;

% Softmax Scatter Plot
subplot(2,2,4);
hold on; grid on;
for i = 1:nC
    idx = (yidxTrue == i);
    scatter(Yprob(1,idx), Yprob(2,idx), 35, colors(i,:), 'filled');
end
xlabel('P(Class 1)'); ylabel('P(Class 2)');
title('Softmax Score Scatter Plot');
legend(cellstr(string(classes)),'Location','best');
hold off;


saveas(fig1, fullfile(PLOT_DIR, ['All_Plots_' timestamp '.png']));




% ROC/EER only 
fig2 = figure;
hold on; grid on;
for i = 1:nC
    scores = Yprob(i,:)';
    labels = (yidxTrue == i);
    [fpr, tpr] = perfcurve(labels, scores, 1);
    fnr = 1 - tpr;
    [~, idx] = min(abs(fpr - fnr));
    eer_point = fpr(idx);

    plot(fpr, tpr, 'Color', colors(i,:), 'LineWidth', 1.5);
    plot(fpr(idx), tpr(idx),'o','MarkerSize',7,'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor','k');
end
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC + EER (Per User)');
legend(legend_labels,'Location','southoutside');
saveas(fig2, fullfile(PLOT_DIR, ['ROC_EER_' timestamp '.png']));

% PCA
fig3 = figure;
gscatter(score(:,1), score(:,2), ytest);
xlabel('PC1'); ylabel('PC2');
title('PCA Scatter Plot');
grid on;
saveas(fig3, fullfile(PLOT_DIR, ['PCA_' timestamp '.png']));

% Softmax scores
fig4 = figure;
hold on; grid on;
for i = 1:nC
    idx = (yidxTrue == i);
    scatter(Yprob(1,idx), Yprob(2,idx), 40, colors(i,:), 'filled');
end
xlabel('P(Class 1)'); ylabel('P(Class 2)');
title('Softmax Score Scatter Plot');
legend(cellstr(string(classes)));
saveas(fig4, fullfile(PLOT_DIR, ['Softmax_' timestamp '.png']));

%Average far & frr with eer curve
avg_FAR = mean(FAR);
avg_FRR = mean(FRR);
EER_value = (avg_FAR + avg_FRR) / 2;

fig5 = figure;
hold on; grid on;

% plot frr & far
plot([0 1], [avg_FAR avg_FAR], 'LineWidth', 2);
plot([0 1], [avg_FRR avg_FRR], 'LineWidth', 2);

% plot eer line
plot([0 1], [EER_value EER_value], '--', 'LineWidth', 2);

% Eer point
plot(0.5, EER_value, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

xlabel('Normalized Threshold');
ylabel('Rate');
title(sprintf('Average FAR/FRR and EER (EER=%.4f)', EER_value));
legend('Avg FAR', 'Avg FRR', 'EER Line', 'EER Point', 'Location', 'best');


saveas(fig5, fullfile(PLOT_DIR, ['Avg_EER_Plot_' timestamp '.png']));

hold off;

%save results
if ~exist('Models','dir'), mkdir('Models'); end
save(fullfile('Models','eval_results.mat'), 'confmat', 'FAR', 'FRR', 'EER', 'acc');

