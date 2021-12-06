function data=CreateData()

     train_data=xlsread('Train_Data.xlsx');
     test_data=xlsread('Test_Data.xlsx');
     load('all_data.mat');

     max_data(1,1)=0;
     min_data(1,1)=0;
     
    % normalize all_data
    [m,n]=size(All_data);
    for i=1:n-1
        max_data(i,1)=max(All_data(:,i));
        min_data(i,1)=min(All_data(:,i));
        for j=1:m
             All_data(j,i)=(All_data(j,i)-min_data(i,1))/(max_data(i,1)-min_data(i,1));        
        end
    end
     
    % normalize Train_Data
    [m,n]=size(train_data);
    for i=1:n-1
        for j=1:m
             train_data(j,i)=(train_data(j,i)-min_data(i,1))/(max_data(i,1)-min_data(i,1));        
        end
    end
    
    
    % normalize Test_Data
    [m,n]=size(test_data);
    for i=1:n-1
        for j=1:m
             test_data(j,i)=(test_data(j,i)-min_data(i,1))/(max_data(i,1)-min_data(i,1));        
        end
    end 
    
    
    % Export
    data.Train=train_data;

    data.Test=test_data;
  
    data.All_data=All_data;

end