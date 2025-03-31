clear
clc
close all
%% Parameter Setting
% rate = load('BFNNRate.mat').rate;
% vec = 30;
% PNR = 5;
% method = "BFNN";"LLM";"ODELSTM";
% v = load(strcat("./precoder/",int2str(PNR), "dB/precoderv",int2str(vec),method,".mat")).vf;
rng(21)

para.N = 256; % the number of transmit antennas
para.K = 1; % user number
para.c = 3e8; % speed of light in free space
para.f = 100e9; % carrier frequency
para.lambda = para.c/para.f;
para.d = para.c/para.f/2; % antenna spacing
para.D = para.d*(para.N-1); % antenna aperture
para.Rayleigh_distance = 2*para.D^2/para.lambda; %Rayleigh_distance 2D^2/Lambda
para.L = 1;  % number of multipaths
V = 100;10:10:100; %velocity 100m/s
para.td = 0.05; %sample time 50ms
%%
methods = [...
    0 ... %LS based BF
    0 ... %
    0 ... %Perfect CSI based BF
    0 ...
    ];
methodName=strvcat( ...
    'LS based BF', ...
    '', ...
    'Perfect CSI based BF', ...
    'LLM' ...
    );
snrdB = 5; 0:2:20;
MonteCalo = 1000;
Tindex = 30; Startindex = 20;
Vlen = length(V);
Rate = zeros(length(snrdB),length(methods));
channelLSMat = zeros(MonteCalo*Vlen,Startindex,para.N);
channelMat = zeros(MonteCalo*Vlen,Tindex-Startindex,para.N);
for vv = 1:Vlen
    for mm = 1:MonteCalo
        x0 = 50+rand*12;
        y0 = 86*(2*rand-1);
        para.pos = [x0, y0];
        dir = sign(2*rand-1); %随机产生运动方向
        channelH = zeros(para.N,Tindex);
        for tt = 1:Tindex
            para.pos = [x0, y0+(tt-1)*V(vv)*para.td*dir];
            channelH(:,tt) = gen_channel(para);
        end
        if(mod(mm,100)==0)
            fprintf([ '\n', 'MonteCalo = %d ', datestr(now), '\n'], mm);
        end
        for ii = 1:length(snrdB)
            SigmaN = db2pow(-snrdB(ii))*para.N;
            noise = (randn(size(channelH))+1j*randn(size(channelH)))/sqrt(2)*sqrt(SigmaN);
            channelHLs = channelH+noise;
            channelHLs0 = fft(eye(para.N))*channelHLs;
            channelLSMat(mm,:,:) = channelHLs0(:, 1:Startindex).';
            channelMat(mm,:,:) = channelH(:,Startindex+1:end).';
            if methods(1) == 1
                for nn = (Startindex+1):Tindex
                    channelHLsBf = sqrt(para.N)*channelHLs(:,nn)/ norm(channelHLs(:,nn));
                    LS_gain = abs(channelH(:,nn)'*channelHLsBf)^2;
                    Rate(ii,1) = Rate(ii,1) + log2(1+LS_gain/SigmaN);
                end
            end
            if methods(2) == 1
                
            end
            %% Perfect CSI based beamforming
            if methods(3) == 1
                for nn = (Startindex+1):Tindex
                    Bfv = sqrt(para.N)*channelH(:,nn)/norm(channelH(:,nn));

                    PerfCSIgain = norm(channelH(:,nn)'*Bfv,'fro')^2;
                    % Rate(ii,3) = Rate(ii,3) + log2(1+PerfCSIgain/SigmaN);
                    Rate(ii,3) = Rate(ii,3) + log2(1+PerfCSIgain/SigmaN);
                end
            end
            if methods(4) == 1
                %% NN based beamforming
                for nn = 1:(Tindex-Startindex)
                    Bfv1 = squeeze(v(mm,nn,:));

                    PerfCSIgain = norm(channelH(:,nn+Startindex)'*Bfv1,'fro')^2;
                    snr = 10^(snrdB(ii)/10);
                    Rate(ii,4) = Rate(ii,4) + log2(1+snr/para.N* PerfCSIgain);
                end

                % Rate(ii,4) = Rate(ii,4) + rate(ii);
            end
        end
    end
end
Rate = Rate/MonteCalo/10;
Rate(:,4) = Rate(:,4)*(Tindex-Startindex);
save train_set\test\5dB\HybridfieldChannel5dB4LLMv100.mat channelMat
save train_set\test\5dB\HybridfieldChannelLS5dB4LLMv100.mat channelLSMat
% save train_set\train\HybridfieldChannel20dB4LLM.mat channelMat
% save train_set\train\HybridfieldChannelLS20dB4LLM.mat channelLSMat

%% plot
fig1 = figure;
vLegend = [];
set(fig1, 'WindowStyle', 'docked');

color = ['g','r','c','b','k','w','m','y'];
line=['--','-','-.',':'];
marker =['o','+','*','x','s','d','^','v','.'];

PlotType = strvcat('-ro', '-b+', '-cx', '-kv','-gs', ...
    '--ro', '--b+', '--cx', '--kv','--gs', ...
    '-.ro','-.b+', '-.cx', '-.kv','-.gs');
for mm=1:length(methods)
    if sum(Rate(:,mm))<=0
        continue;
    end
    vLegend = strvcat(vLegend,  methodName(mm,:));
    plot(snrdB, Rate(:,mm), PlotType(mm,:),'LineWidth',1.5,'MarkerSize',6);
    hold on;
end
xlabel('SNR [dB]');
ylabel('Mutual Information(bit/s/Hz)');
grid on;
legend(vLegend,'Location','NorthWest')
xlim([snrdB(1)-5 snrdB(end)+5])



