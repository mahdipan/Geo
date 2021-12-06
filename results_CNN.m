%% Load dataset
data=CreateData();
TrainData = data.Train;
TestData = data.All_data;
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


result_all_data = double(predict(net,TestInput)');
result_all_data=result_all_data';
xlswrite('result_cnn_1.xlsx',result_all_data(1:1000000,:));
xlswrite('result_cnn_2.xlsx',result_all_data(1000001:2000000,:));
xlswrite('result_cnn_3.xlsx',result_all_data(2000001:3000000,:));
xlswrite('result_cnn_4.xlsx',result_all_data(3000001:end,:));