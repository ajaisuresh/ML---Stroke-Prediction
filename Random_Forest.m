clc
clear all
close all
warning off

data = readtable("stroke_prediction.csv");
% X --> age, BMI, avg_glucose_level
X = data(:, [3 9 10]);
Y = data(:, 12);

% Random Forest is known as TreeBagger in MathsWork 
rf_mdl = TreeBagger(500, X, Y, method = 'classification', OOBPrediction= 'on');
err = mean(oobError(rf_mdl));
yy = table2array(Y);
y2hat = str2num(cell2mat(predict(rf_mdl,X)));
cm_rf = confusionmat(yy,y2hat)
confusionchart(yy,y2hat)


tp_rf = cm_rf(1,1);
fp_rf = cm_rf(1,2);
fn_rf = cm_rf(2,1);
tn_rf = cm_rf(2,2);

% Accuracy score
ac_rf = ((tp_rf + tn_rf)/(tp_rf+fp_rf+fn_rf+tn_rf));
disp('The accuracy score is')
disp(ac_rf)

% Precision score
pc_rf = (tp_rf/(tp_rf + fp_rf));
disp('The precision score is')
disp(pc_rf)

%Recall score
rc_rf= (tp_rf/(tp_rf+fn_rf));
disp('The recall score is')
disp(rc_rf)

% Specifity score
sc_rf = (tn_rf/(tn_rf+fp_rf));
disp('The specifity score is')
disp(sc_rf)

% F1 Score
f1_score_rf = 2*(rc_rf*pc_rf)/(rc_rf+pc_rf);
disp('The F1 score is')
disp(f1_score_rf)
