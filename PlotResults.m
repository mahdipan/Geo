%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML120
% Project Title: Time-Series Prediction using GMDH
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function PlotResults(Targets, Outputs, Title)

    Errors = Targets - Outputs;
    MSE = mean(Errors.^2);
    RMSE = sqrt(MSE);
    [~,~,R] = postreg(Targets,Outputs,'hide');
    
    figure;
    plot(Targets);
    hold on;
    plot(Outputs);
    legend('Targets','Outputs');
    title(Title);

    figure;
    t = 0:0.1:1;
    plot(t,t,'k','linewidth',2)
    hold on
    plot(Outputs,Targets,'r.','markersize',15)
    hold off
    title(['R = ' num2str(R)]);
    ylabel('Target');
    xlabel('Output')
    title(Title);    

end