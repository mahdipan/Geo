

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

%  MIN = min(TrainData,[],1);
%  MAX = max(TrainData,[],1);
%  TrainData = (TrainData-MIN)./(MAX-MIN);
%  TestData = (TestData-MIN)./(MAX-MIN);

%% Partitioning

Ntr = size(TrainData,1);
Valtr = round(0.1*Ntr);
TrainInputs = TrainData(1:end-Valtr,1:end-1);
TrainTargets = TrainData(1:end-Valtr,end);
ValInputs = TrainData(Ntr-Valtr+1:end,1:end-1);
ValTargets = TrainData(Ntr-Valtr+1:end,end);
TestInputs = TestData(:,1:end-1);
TestTargets = TestData(:,end);

%% Reshape Data from N*6 to 6*1*1*N

%### Train
VectorSize = size(TrainInputs,2); % 6
DataNumtr = size(TrainInputs,1);
InputSizetr = [VectorSize,1,1,DataNumtr];
temp = TrainInputs';
TrainInput = reshape(temp,InputSizetr);

%### Validation
VectorSize = size(ValInputs,2); % 6
DataNumval = size(ValInputs,1);
InputSizeval = [VectorSize,1,1,DataNumval];
temp = ValInputs';
ValInput = reshape(temp,InputSizeval);

%### Test
VectorSize = size(TestInputs,2); % 6
DataNumts = size(TestInputs,1);
InputSizets = [VectorSize,1,1,DataNumts];
temp = TestInputs';
TestInput = reshape(temp,InputSizets);

%% Define Network Architecture

% Define the convolutional neural network architecture.
tic;
layers = [
    imageInputLayer([VectorSize 1] ,'Normalization', 'none') % 22X1X1 refers to number of features per sample
    convolution2dLayer([14,1],30,'Stride',1,'Padding','same');
    batchNormalizationLayer
    reluLayer  
    dropoutLayer    
    averagePooling2dLayer([3,1],'Stride',2); 
    
   
 

    
%     convolution2dLayer(7,48,'Stride',1,'Padding',3);
%     reluLayer();
%     maxPooling2dLayer(2,'Stride',2);
    
    
%     convolution2dLayer([12,1],3,'Padding','same')
%     batchNormalizationLayer
%     reluLayer  
%     
%     maxPooling2dLayer([2,1])
%     
%     convolution2dLayer([8,1],10,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer

    fullyConnectedLayer(10) % 20 refers to number of neurons in next FC hidden layer?
    fullyConnectedLayer(1) % OutputSize refers to number of neurons in next output layer (number of output classes)
    regressionLayer];   

%% Train Network

options = trainingOptions('sgdm',...
    'InitialLearnRate',0.01,...
    'MaxEpochs',100, ...
    'MiniBatchSize',8,...
    'ValidationFrequency',4, ...,
    'ValidationPatience',15,...
    'ValidationData',{ValInput,ValTargets},...
    'Plots','training-progress',...
    'ExecutionEnvironment', 'cpu');

net = trainNetwork(TrainInput,TrainTargets,layers,options);
time = toc;

%% Evaluate Network

% on train
TrainOutputs = double(predict(net,TrainInput)');
Title = 'Result (train)';
PlotResults(TrainTargets, TrainOutputs', Title)

% on test
TestOutputs = double(predict(net,TestInput)');
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

% writetable(Table,'CNNResults.xlsx')

