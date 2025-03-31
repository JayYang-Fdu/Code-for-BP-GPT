function [H] = gen_channel(para)

lmd = 1/para.lambda;
Drayl = para.Rayleigh_distance; %瑞丽距离
pos = para.pos; %[x,y]
x = pos(1); %计算x-axis
y = pos(2); %计算y-axis
rd = sqrt(x^2+y^2); %计算用户与基站之间的距离
theta = atan(y/x);  %计算角度


if rd < Drayl %近场
    beta = 1/(4*pi*lmd*rd)*exp(-1i * 2 * pi* lmd*rd);
    n = ((-(para.N-1)/2 : (para.N-1)/2) * para.d)';
    r = sqrt(rd^2 + n.^2 - 2*rd*n*cos(theta))-rd;
    a = exp(-1i * 2 * pi * lmd .* r );
    Htmp = beta*exp(-1i * 2 * pi* lmd*rd)*a;
    NormHf = norm(Htmp,'fro')^2;
    c = sqrt(para.N/NormHf);
    H = c*Htmp;
else %远场
    beta =  ((1/(4*pi*5*lmd))^2)*(5/rd);
    a = exp(1i*pi*(-(para.N-1)/2 : (para.N-1)/2)'*cos(theta));
    Htmp = beta*a;
    NormHf = norm(Htmp,'fro')^2;
    c = sqrt(para.N/NormHf);
    H = c*Htmp;
end

end