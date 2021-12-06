

%% Load dataset
data=CreateData();
TrainData = data.Train;
TestData = data.All_data;

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

result_all_data = double(predict(net,testInput)');
result_all_data=result_all_data';
xlswrite('result_LSTM_1.xlsx',result_all_data(1:1000000,:));
xlswrite('result_LSTM_2.xlsx',result_all_data(1000001:2000000,:));
xlswrite('result_LSTm_3.xlsx',result_all_data(2000001:3000000,:));
xlswrite('result_LSTM_4.xlsx',result_all_data(3000001:end,:));


