
function Eval = Evaluate(Yreal, YN)
N = length(Yreal);
SSE = sum((Yreal-YN).^2);
MSE = mse(YN - Yreal);
RMSE = sqrt(MSE);
MAE = mae((YN - Yreal));
[~,~,R] = postreg(YN,Yreal,'hide');
Eval = [SSE, MSE, RMSE, MAE, R];
end