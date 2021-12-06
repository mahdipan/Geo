




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

TrainInputs = TrainData(:,1:end-1);
TrainTargets = TrainData(:,end);
TestInputs = data.All_data(:,1:end-1);
TestTargets = data.All_data(:,end);

%% Reshape Data 

%### Train
trainInput = cell(1,size(TrainInputs,1));
for i=1:size(TrainInputs,1)
    trainInput{i} = TrainInputs(i,:)';
end
TrainTarget = cell(1,size(TrainTargets,1));
for i=1:size(TrainTargets,1)
    TrainTarget{i} = TrainTargets(i,:)';
end

%### Test
testInput = cell(1,size(TestInputs,1));
for i=1:size(TestInputs,1)
    testInput{i} = TestInputs(i,:)';
end
TestTarget = cell(1,size(TestTargets,1));
for i=1:size(TestTargets,1)
    TestTarget{i} = TestTargets(i,:)';
end



%% Evaluate RNN(RMSE)

% on train
[x,xi,ai,t] = preparets(net,trainInput,{},TrainTarget);
TrainOutput = net(x,xi,ai);
TrainOutputs = zeros(size(TrainOutput))';
for i=1:size(TrainOutputs,1)
    TrainOutputs(i) = TrainOutput{i};
end
%Title = 'Result (train)';
%PlotResults(TrainTargets(3:end), TrainOutputs, Title)

% on test
[x,xi,ai,t] = preparets(net,testInput,{},TestTarget);
TestOutput = net(x,xi,ai);
TestOutputs = zeros(size(TestOutput))';
for i=1:size(TestOutputs,1)
    TestOutputs(i) = TestOutput{i};
end
%Title = 'Result (test)';
%PlotResults(TestTargets(3:end), TestOutputs, Title)

%% Evaluation Metrics

Eval2 = Evaluate(TrainTargets(3:end), TrainOutputs);      %% Train
Eval3 = Evaluate(TestTargets(3:end), TestOutputs);      %% Test

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
xlswrite('result_RNN_1.xlsx',TestOutputs(1:1000000,:));
xlswrite('result_RNN_2.xlsx',TestOutputs(1000001:2000000,:));
xlswrite('result_RNN_3.xlsx',TestOutputs(2000001:3000000,:));
xlswrite('result_RNN_4.xlsx',TestOutputs(3000001:end,:));



