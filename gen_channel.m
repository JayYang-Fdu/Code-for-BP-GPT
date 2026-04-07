function [channelH] = gen_channel(para)

channelH = zeros(para.N, para.K);
lmd = 1/para.lambda;
Drayl = para.Rayleigh_distance; %瑞丽距离

pos = para.pos; %[x,y] for K users
alpha = 0.1; % Nlos 的功率相对于LoS为-10dB


for kk = 1:para.K
    L = randi(para.L);
    x = pos(kk,1); %计算x-axis
    y = pos(kk,2); %计算y-axis
    rd = sqrt(x^2+y^2); %计算第k个用户与基站之间的距离：Los
    theta = atan(y/x);  %计算角度
    phi = rand*2*pi;

    %% 生成Los径
    if rd < Drayl %近场
        beta = 1/(4*pi*lmd*rd)*exp(-1i * 2 * pi* lmd*rd)*exp(1j*phi);
        n = ((-(para.N-1)/2 : (para.N-1)/2) * para.d)';
        r = sqrt(rd^2 + n.^2 - 2*rd*n*cos(theta))-rd;
        a = exp(-1i * 2 * pi * lmd .* r );
        Htmp = beta*exp(-1i * 2 * pi* lmd*rd)*a;
        % NormHf = norm(Htmp,'fro')^2;
        % c = sqrt(para.N/NormHf);
        % H = c*Htmp;
    else %远场
        beta =  ((1/(4*pi*5*lmd))^2)*(5/rd)*exp(1j*phi);
        a = exp(1i*pi*(-(para.N-1)/2 : (para.N-1)/2)'*cos(theta));
        Htmp = beta*a;

    end

    %% 生成NLos径信道,假设LoS/NLoS=30dB
    thetaV = pi/3*(2*rand(L-1,1)-1);
    phiV = rand(L-1,1)*2*pi;
    rdV = 20+rand(L-1,1)*80;
    for ll = 1:L-1
        if rdV(ll)< Drayl
            betaNlos = 1/(4*pi*lmd*rdV(ll))*exp(-1i * 2 * pi* lmd*rdV(ll))*exp(1j*phiV(ll));
            n = ((-(para.N-1)/2 : (para.N-1)/2) * para.d)';
            r = sqrt(rdV(ll)^2 + n.^2 - 2*rdV(ll)*n*cos(thetaV(ll)))-rdV(ll);
            aNlos = exp(-1i * 2 * pi * lmd .* r );
            HtmpNlosTmp =  betaNlos*exp(-1i * 2 * pi* lmd*rdV(ll))*aNlos;
            Htmp = Htmp + HtmpNlosTmp*alpha; 
        else %远场
            betaNlos =  ((1/(4*pi*5*lmd))^2)*(5/rd)*exp(1j*phiV(ll));
            a = exp(1i*pi*(-(para.N-1)/2 : (para.N-1)/2)'*cos(thetaV(ll)));
            HtmpNlosTmp = betaNlos*a;
            Htmp = Htmp + HtmpNlosTmp*alpha;
        end
    end
    %%
    NormH= norm(Htmp,'fro')^2;
    c = sqrt(para.N/NormH);
    H = c*Htmp;
    channelH(:,kk) = H;
end
% channelH = channelH.';
end