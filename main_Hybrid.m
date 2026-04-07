clear
clc
close all
%% Parameter Setting
% rate = load('NNRate.mat').rate;
para.N = 256; % the number of transmit antennas
para.K = 1; % user number
para.c = 3e8; % speed of light in free space
para.f = 100e9; % carrier frequency
para.lambda = para.c/para.f;
para.d = para.c/para.f/2; % antenna spacing
para.D = para.d*(para.N-1); % antenna aperture
para.Rayleigh_distance = 2*para.D^2/para.lambda; %Rayleigh_distance 2D^2/Lambda
para.L = 10;  % number of multipaths
V = 100; 10:10:100;  %velocity 100m/s
para.td = 0.05; %sample time 50ms
%%
methods = [...
    0 ... %LS based BF
    1 ... %far field based BF
    0 ... %Perfect CSI based BF
    1 ...
    0 ...
    0 ...
    ];
methodName=strvcat( ...
    'LS based BF', ...
    'far field based BF', ...
    'Perfect CSI based BF', ...
    'LLM', ...
    'PAD' , ...
    'Kalman' ...
    );
snrdB =0; 0:2:20;
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
        dir = sign(2*rand-1); %随机产生运动方向，目前假定是+1，-1
        channelH = zeros(para.N,Tindex);
        
        for tt = 1:Tindex
            para.pos = [x0, y0+(tt-1)*V(vv)*para.td*dir];
            channelH(:,tt) = gen_channel(para);
            
        end
        if(mod(mm,100)==0)
            fprintf([ '\n', 'MonteCalo = %d ', datestr(now), '\n'], mm);
        end
        for ii = 1:length(snrdB)
            snr = 10^(snrdB(ii)/10);
            SigmaN = db2pow(-snrdB(ii));
            noise = (randn(size(channelH))+1j*randn(size(channelH)))/sqrt(2)*sqrt(SigmaN);
            channelHLs = channelH+noise;
            channelHLs0 = fft(eye(para.N))*channelHLs;
            channelLSMat((vv-1)*MonteCalo + mm,:,:) = channelHLs0(:, 1:Startindex).';
            channelMat((vv-1)*MonteCalo + mm,:,:) = channelH(:,Startindex+1:end).';
            if methods(1) == 1
                for nn = Startindex+1:Tindex
                    channelHLsBf = sqrt(para.N)*channelHLs(:,nn)/ norm(channelHLs(:,nn));
                    LS_gain = abs(channelH(:,nn)'*channelHLsBf)^2;
                    Rate(ii,1) = Rate(ii,1) + log2(1+LS_gain/SigmaN);
                end
            end
            if methods(2) == 1
                delta_angle = 32;
                t = (-(delta_angle-1)/2:1:(delta_angle/2))*(2/delta_angle);
                g = 0:para.N-1;
                A = exp(-1j * pi * g' * t);
                y = abs(channelH' * A);
                [value,pos] = max(y.');
                for nn = Startindex+1:Tindex
                    at = A(:,pos(nn));
                    y_far = channelH(:,nn)' *at;
                    far_field_rate_gain = norm(y_far)^2;
                    Rate(ii,2) = Rate(ii,2) + log2(1+snr/para.N*far_field_rate_gain);
                end

            end
            %% Perfect CSI based beamforming
            if methods(3) == 1
                for nn = Startindex+1:Tindex
                    Bfv = exp(1j*angle(channelH(:,nn)));
                    PerfCSIgain = norm(channelH(:,nn)'*Bfv,'fro')^2;
                    Rate(ii,3) = Rate(ii,3) + log2(1+snr/para.N*PerfCSIgain);
                end
            end
            if methods(4) == 1
                %% Perfect CSI based beamforming
                Rate(ii,4) = Rate(ii,4) + rate(ii);
            end
            if methods(5) == 1
                fftMat = fft(eye(para.N))/sqrt(para.N);
                HLs0 = fftMat' * channelH;
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
                for nn = 1:Ns
                    % index = find(HLs0(:,ll)==HLsdes(nn,ll));
                    % indexMat(nn,ll) = index;
                    % h = h+ HLsdes(nn,ll)*fftMat(:,index);
                    G = hankel(Hused(nn,1:end-preT), Hused(nn,preT:end-1));
                    g = Hused(nn,preT+1:end).';
                    p = -pinv(G)*g;
                    gpre = Hused(nn,preT+1:end);
                    for tt = 1:preT
                        ghat = -gpre*p;
                        gpre = [gpre(2:end),ghat];
                    end
                    Hpre = Hpre + fftMat(:,nn)*gpre;
                end
                for nn = 1:Tindex-Startindex
                    Bfv = exp(1j*angle(Hpre(:,nn)));

                    PerfCSIgain = norm(channelH(:,nn)'*Bfv,'fro')^2;
                    Rate(ii,5) = Rate(ii,5) + log2(1+PerfCSIgain/SigmaN);
                end
            end
            if methods(6) == 1
               
                
            end
        end
    end
end
Rate = Rate/MonteCalo/10;
% Rate(:,4) = Rate(:,4)*(Tindex-Startindex);
save train_set\test\multipath\HybridfieldChannel0dB4LLMv100.mat channelMat
save train_set\test\multipath\HybridfieldChannelLS0dB4LLMv100.mat channelLSMat
% save train_set\train\HybridfieldChannel10dB4LLM.mat channelMat
% save train_set\train\HybridfieldChannelLS10dB4LLM.mat channelLSMat

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



