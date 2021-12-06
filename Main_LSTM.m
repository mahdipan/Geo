
%% Main CNN

close all;
clear;
clc;
warning off;

%% Load dataset
data=CreateData();
TrainData = data.Train;
TestData = data.Test;
All_data = data.All_data;

%% Normalization
% 
% MIN = min(TrainData,[],1);
% MAX = max(TrainData,[],1);
% TrainData = (TrainData-MIN)./(MAX-MIN);
% TestData = (TestData-MIN)./(MAX-MIN);

%% Partitioning

Ntr = size(TrainData,1);
Valtr = round(0.1*Ntr);
TrainInputs = TrainData(1:end-Valtr,1:end-1);
TrainTargets = TrainData(1:end-Valtr,end);
ValInputs = TrainData(Ntr-Valtr+1:end,1:end-1);
ValTargets = TrainData(Ntr-Valtr+1:end,end);
TestInputs = TestData(:,1:end-1);
TestTargets = TestData(:,end);

%% Reshape Data 

%### Train
trainInput = cell(size(TrainInputs,1),1);
for i=1:size(TrainInputs,1)
    trainInput{i} = TrainInputs(i,:)';
end

%### Validation
valInput = cell(size(ValInputs,1),1);
for i=1:size(ValInputs,1)
    valInput{i} = ValInputs(i,:)';
end

%### Test
testInput = cell(size(TestInputs,1),1);
for i=1:size(TestInputs,1)
    testInput{i} = TestInputs(i,:)';
end


%% Define Network Architecture

VectorSize = size(TrainInputs,2);
numHiddenUnits1 = 100;
tic;
layers = [ ...
    sequenceInputLayer(VectorSize)
    bilstmLayer(numHiddenUnits1,'OutputMode','last')
    dropoutLayer(0.5);
    fullyConnectedLayer(10)
    fullyConnectedLayer(1)
    
    regressionLayer]; 

%% Train Network

miniBatchSize = 10;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',1000, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationFrequency',20, ...
    'ValidationPatience',25,...
    'Shuffle','every-epoch', ...
    'Verbose',0, ...
    'Plots','training-progress',...
    'ValidationData',{valInput,ValTargets});

net = trainNetwork(trainInput,TrainTargets,layers,options);
time = toc;

%% Evaluate Network

% on train
TrainOutputs = double(predict(net,trainInput)');
Title = 'Result (train)';
PlotResults(TrainTargets, TrainOutputs', Title)

% on test
TestOutputs = double(predict(net,testInput)');
Title = 'Result (test)';
PlotResults(TestTargets, TestOutputs', Title)

%% Evaluation Metrics

Eval2 = Evaluate(TrainTargets, TrainOutputs');      %% Train
Eval3 = Evaluate(TestTargets, TestOutputs');      %% Test

SSETR = Eval2(1);
MSETR = Eval2(2);
RMSETR = Eval2(3);
MAETR = Eval2(4);
RTR = Eval2(5);

SSETS = Eval3(1);
MSETS = Eval3(2);
RMSETS = Eval3(3);
MAETS = Eval3(4);
RTS = Eval3(5);

Table = table(SSETR, MSETR, RMSETR, MAETR, RTR, ...
 SSETS, MSETS, RMSETS, MAETS, RTS);
disp(Table)

% writetable(Table,'LSTMResults.xlsx')

