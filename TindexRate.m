clear
clc
% close all
%% Parameter Setting
% rate = load('BFNNRate.mat').rate;
vec = 30;
PNR = 0;

Bfv = load(strcat("./precoder/distill/precoderv",int2str(vec),"LLM.mat")).vf;
BfvODELSTM = load(strcat("./precoder/precoderv",int2str(vec),"ODELSTM.mat")).vf;
BfvNN = load(strcat("./precoder/precoderv",int2str(vec),"BFNN.mat")).vf;
% channelLs = load(strcat("./train_set/test/HybridfieldChannelLS",int2str(PNR),"dB4LLMv",int2str(vec),".mat")).channelLSMat;
channel = load(strcat("train_set\test\multipath\HybridfieldChannel",int2str(PNR),"dB4LLMv",int2str(vec),".mat")).channelMat;
rng(21)

para.N = 256; % the number of transmit antennas
para.K = 1; % user number
para.c = 3e8; % speed of light in free space
para.f = 100e9; % carrier frequency
para.lambda = para.c/para.f;
para.d = para.c/para.f/2; % antenna spacing
para.D = para.d*(para.N-1); % antenna aperture
para.Rayleigh_distance = 2*para.D^2/para.lambda; %Rayleigh_distance 2D^2/Lambda
para.L = 10;  % number of multipaths
V = 30;10:10:100; %velocity 100m/s
para.td = 0.05; %sample time 20ms
%%
methods = [...
    0 ... %LS based BF
    1 ... %far field based BF
    1 ... %Perfect CSI based BF
    1 ... % LLM
    0 ... % BFNN
    1 ... % LSTM
    0 ... PAD
    ];
methodName=strvcat( ...
    'LS based BF', ...
    'Far Field based BF', ...
    'Perfect CSI based BF', ...
    'BP-GPT' , ...
    'BFNN' , ...
    'LSTM', ...
    'PAD Channel Prediction' ...
    );
snrdB = 15;
MonteCalo = 1000;
Tindex = 30; Startindex = 20;
Vlen = length(V);
Rate = zeros(Tindex-Startindex,length(methods));
channelLSMat = zeros(MonteCalo*Vlen,Startindex,para.N);
channelMat = zeros(MonteCalo*Vlen,Tindex-Startindex,para.N);
for vv = 1:Vlen


    SigmaN = db2pow(-snrdB);
    snr = 10^(snrdB/10);
    alpha = db2pow(-PNR)*para.N;
    for nn = 1:(Tindex-Startindex)
        for mm = 1:MonteCalo
            x0 = 50+rand*12;
            y0 = 86*(2*rand-1);
            para.pos = [x0, y0];
            dir = sign(2*rand-1); %随机产生运动方向，目前假定是+1，-1
            % H = zeros(para.N,Tindex);
            % for tt = 1:Tindex
            %     para.pos = [x0, y0+(tt-1)*V(vv)*para.td*dir];
            %     H(:,tt) = gen_channel(para);
            % end
            % channelH = H(:,20+nn);
            channelH = squeeze(channel(mm,:,:)).';



            if methods(1) == 1

            end
            if methods(2) == 1
                delta_angle = 32;
                t = (-(delta_angle-1)/2:1:(delta_angle/2))*(2/delta_angle);
                g = 0:para.N-1;
                A = exp(-1j * pi * g' * t);
                y = abs(channelH(:,nn)' * A);
                [value,pos] = max(y.');
                
                at = A(:,pos);
                y_far = channelH(:,nn)' *at;
                far_field_rate_gain = norm(y_far)^2;

                Rate(nn,2) = Rate(nn,2) + log2(1+snr/para.N*far_field_rate_gain);
            end

            if methods(3) == 1
                BfvCSI = sqrt(para.N)*channelH(:,nn)/norm(channelH(:,nn));

                PerfCSIgain = norm(channelH(:,nn)'*BfvCSI,'fro')^2;
                % Rate(ii,3) = Rate(ii,3) + log2(1+PerfCSIgain/SigmaN);
                Rate(nn,3) = Rate(nn,3) + log2(1+snr/para.N* PerfCSIgain);
            end
            if methods(4) == 1
                %% LLM based beamforming
                BfvLLM = squeeze(Bfv(mm, nn,:));
                PerfCSIgainLLM = abs(channelH(:,nn)'*BfvLLM)^2;

                Rate(nn,4) = Rate(nn,4) + log2(1+snr/para.N* PerfCSIgainLLM);

                % Rate(ii,4) = Rate(ii,4) + rate(ii);
            end
            if methods(5) == 1
                %% BFNN based beamforming

                BfvBFNN = squeeze(BfvNN(mm,nn,:));

                PerfCSIgainNN = norm(channelH'*BfvBFNN,'fro')^2;
                % snr = 10^(snrdB/10);
                Rate(nn,5) = Rate(nn,5) + log2(1+snr/para.N* PerfCSIgainNN);
            end
            if methods(6) == 1
                %% LSTM based beamforming

                BfvLSTM = squeeze(BfvODELSTM(mm,nn,:));

                PerfCSIgainLSTM = norm(channelH'*BfvLSTM,'fro')^2;
                % snr = 10^(snrdB/10);
                Rate(nn,6) = Rate(nn,6) + log2(1+snr/para.N* PerfCSIgainLSTM);
            end
            if methods(7) == 1
                fftMat = fft(eye(para.N))/sqrt(para.N);
                HLs0 = fftMat' * H;
                HLsdes = sort(HLs0,'descend');
                sumP = sum(sum(abs(HLs0).^2));
                gamma = 0.98;
                for n = 1:para.N
                    if sum(sum(abs(HLsdes(1:n,:)).^2)) >= gamma*sumP
                        break
                    end
                end
                Ns = n;
                indexMat = zeros(Ns,Startindex);
                Hused = HLsdes(:,1:Startindex);
                preT = Tindex-Startindex;
                Hpre = 0;
                for ss = 1:Ns
                    % index = find(HLs0(:,ll)==HLsdes(nn,ll));
                    % indexMat(nn,ll) = index;
                    % h = h+ HLsdes(nn,ll)*fftMat(:,index);
                    G = hankel(Hused(ss,1:end-preT), Hused(ss,preT:end-1));
                    g = Hused(ss,preT+1:end).';
                    p = -pinv(G)*g;
                    gpre = Hused(ss,preT+1:end);
                    for tt = 1:preT
                        ghat = -gpre*p;
                        gpre = [gpre(2:end),ghat];
                    end
                    Hpre = Hpre + fftMat(:,ss)*gpre;
                end
                Bfv = exp(1j*angle(Hpre(:,nn)));

                PerfCSIgain = norm(channelH'*Bfv,'fro')^2;
                Rate(nn,7) = Rate(nn,7) + log2(1+PerfCSIgain/SigmaN);
            end

        end
    end
end
Rate = Rate/MonteCalo;
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
    plot(1:10, Rate(:,mm), PlotType(mm,:),'LineWidth',1.5,'MarkerSize',6);
    hold on;
end
xlabel('Timestamp-index');
ylabel('Average SE (bits/s/Hz)');
title(strcat('PNR=',int2str(PNR),'dB,v=',int2str(vec),'m/s'))
grid on;
legend(vLegend,'Location','NorthWest')
xlim([1 10])



