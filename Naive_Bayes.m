clc
clear all
close all
warning off

data = readtable("stroke_prediction.csv");
% The set of predictor vairables for the optimal model 
X = data(:, [3 4 5 9 10]);
Y = data(:, 12);

% Set the seeds of the random number generators to ensure reproducability
rng(1);
tallrng('default')

% Train a Naive Bayes classifier
NB_mdl = fitcnb(X,Y);

% use the crossval() function to return cross-validated (partitioned)
% machine learning model from a trainedmodel. By default, crossval() uses
% 10-fold cross-validation on the trained data. 

CV_NB_mdl = crossval(NB_mdl);
yhat = kfoldPredict(CV_NB_mdl);
yy = table2array(Y);
cm_nb = confusionmat(yy, yhat)
confusionchart(yy,yhat)

% Confusion matrix
tp_nb = cm_nb(1,1);
fp_nb = cm_nb(1,2);
fn_nb = cm_nb(2,1);
tn_nb = cm_nb(2,2);

% Accuracy score
ac_nb = ((tp_nb + tn_nb)/(tp_nb+fp_nb+fn_nb+tn_nb));
disp('The accuracy score is')
disp(ac_nb)

% Precision score
pc_nb = (tp_nb / (tp_nb + fp_nb));
disp('The precision score is')
disp(pc_nb)

%Recall score
rc_nb = (tp_nb / (tp_nb + fn_nb));
disp('The recall score is')
disp(rc_nb)

% Specifity score
sc_nb = (tn_nb / (tn_nb + fp_nb));
disp('The specifity score is')
disp(sc_nb)

% F1 Score
f1_score_nb = 2 * (rc_nb *pc_nb) / (rc_nb + pc_nb);
disp('The F1 score is')
disp(f1_score_nb)