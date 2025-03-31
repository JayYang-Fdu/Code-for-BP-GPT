clear
clc
% close all
%% Parameter Setting
% rate = load('BFNNRate.mat').rate;
PNR = 10;
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
V = 10:10:100; %velocity 100m/s
para.td = 0.05; %sample time 20ms
%%
methods = [...
    1 ... %LS based BF
    0 ... %
    1 ... %Perfect CSI based BF
    1 ... % LLM
    1 ... % BFNN
    1 ... % LSTM
    ];
methodName=strvcat( ...
    'LS based BF', ...
    '', ...
    'Perfect CSI based BF', ...
    'BP-GPT' , ...
    'BFNN' , ...
    'LSTM' ...
    );
snrdB = 10;
MonteCalo = 1000;
Tindex = 30; Startindex = 20;
Vlen = length(V);
Rate = zeros(100,length(methods));
channelLSMat = zeros(MonteCalo*Vlen,Startindex,para.N);
channelMat = zeros(MonteCalo*Vlen,Tindex-Startindex,para.N);
for vv = 1:Vlen
    vv
    vec = V(vv);


    Bfv = load(strcat("./precoder/",int2str(PNR), "dB/precoderv",int2str(vec),"LLM.mat")).vf;
    BfvODELSTM = load(strcat("./precoder/",int2str(PNR), "dB/precoderv",int2str(vec),"ODELSTM.mat")).vf;
    BfvNN = load(strcat("./precoder/",int2str(PNR), "dB/precoderv",int2str(vec),"BFNN.mat")).vf;
    % channelLs = load(strcat("./train_set/test/HybridfieldChannelLS",int2str(PNR),"dB4LLMv",int2str(vec),".mat")).channelLSMat;
    channel = load(strcat("train_set\test\",int2str(PNR),"dB\HybridfieldChannel",int2str(PNR),"dB4LLMv",int2str(vec),".mat")).channelMat;

    for mm = 1:MonteCalo

        % x0 = 50+rand*12;
        % y0 = 86*(2*rand-1);
        % para.pos = [x0, y0];
        % dir = sign(2*rand-1); %随机产生运动方向，目前假定是+1，-1
        % channelH = zeros(para.N,Tindex);
        % for tt = 1:Tindex
        %     para.pos = [x0, y0+(tt-1)*V(vv)*para.td*dir];
        %     channelH(:,tt) = gen_channel(para);
        % end

        channelH = squeeze(channel(mm,:,:)).';
        SigmaN = db2pow(-snrdB)*para.N;
        alpha = db2pow(-PNR)*para.N;
        if methods(1) == 1
            noise = (randn(size(channelH))+1j*randn(size(channelH)))/sqrt(2)*sqrt(db2pow(alpha));
            channelHLs = channelH+noise;
            for nn = 1:(Tindex-Startindex)
                channelHLsBf = sqrt(para.N)*channelHLs(:,nn)/ norm(channelHLs(:,nn));
                LS_gain = abs(channelH(:,nn)'*channelHLsBf)^2;
                Rate(vec,1) = Rate(vec,1) + log2(1+LS_gain/SigmaN);
            end
        end
        if methods(2) == 1
            

        end

        if methods(3) == 1
            for nn = 1:(Tindex-Startindex)
                BfvCSI = sqrt(para.N)*channelH(:,nn)/norm(channelH(:,nn));

                PerfCSIgain = norm(channelH(:,nn)'*BfvCSI,'fro')^2;
                % Rate(ii,3) = Rate(ii,3) + log2(1+PerfCSIgain/SigmaN);
                Rate(vec,3) = Rate(vec,3) + log2(1+PerfCSIgain/SigmaN);
            end
        end
        if methods(4) == 1
            %% LLM based beamforming
            for nn = 1:(Tindex-Startindex)

                BfvLLM = squeeze(Bfv(mm, nn,:));

                PerfCSIgainLLM = norm(channelH(:,nn)'*BfvLLM,'fro')^2;
                snr = 10^(snrdB/10);
                Rate(vec,4) = Rate(vec,4) + log2(1+snr/para.N* PerfCSIgainLLM);
            end
        end
        if methods(5) == 1
            %% BFNN based beamforming

            for nn = 1:(Tindex-Startindex)

                BfvBFNN = squeeze(BfvNN(mm,nn,:));

                PerfCSIgainNN = norm(channelH(:,nn)'*BfvBFNN,'fro')^2;
                snr = 10^(snrdB/10);
                Rate(vec,5) = Rate(vec,5) + log2(1+snr/para.N* PerfCSIgainNN);
            end
        end
        if methods(6) == 1
            %% LSTM based beamforming

            for nn = 1:(Tindex-Startindex)

                BfvLSTM = squeeze(BfvODELSTM(mm,nn,:));

                PerfCSIgainLSTM = norm(channelH(:,nn)'*BfvLSTM,'fro')^2;
                snr = 10^(snrdB/10);
                Rate(vec,6) = Rate(vec,6) + log2(1+snr/para.N* PerfCSIgainLSTM);
            end
        end

    end

end
Rate = Rate/MonteCalo/10;
% Rate(:,4) = Rate(:,4)*(Tindex-Startindex);
% save train_set\test\HybridfieldChannel5dB4LLM.mat channelMat
% save train_set\test\HybridfieldChannelLS5dB4LLM.mat channelLSMat
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
    plot(10:10:100, Rate(10:10:100,mm), PlotType(mm,:),'LineWidth',1.5,'MarkerSize',6);
    hold on;
end
xlabel('Velocity');
ylabel('Average SE (bits/s/Hz)');
title(strcat('PNR=',int2str(PNR),'dB,v=',int2str(vec),'m/s'))
grid on;
legend(vLegend,'Location','NorthWest')
xlim([10 100])



