% ### CreateSOAElikeWF2.m ###      2023.06.14

% --- Purpose
% o Different spectral pattern re CreateSOAElikeWF.m (see notes below)
% o Create time waveform with various noisy "peaks" so to create a
% ground truth file to plug into xc codes (e.g., RSinterpeakCorr3.m)
% o individual noisy sinusoid waveforms are crafted via funcNoisySin.m (see
% bottom of code for details re input params) and then added up together

% To save and set up for extractSOAEenvelope.m (and thereby
% RSinterpeakCorr3.m), save the computed summed waveform as follows:
% > save('***.mat','wf')

% --- NOTES (specific to CreateSOAElikeWF2.m)
% o 

clear
% ==============================================
% --- base wf params 
In.length= 60;   % length of total generated waveform(s) [s]
In.SR= 44100;      % sample rate [Hz]
In.L= 8192;        % length of shorter time segments for spec. averaging
P.quant= 0;        % quantize the peak freqs. re the total window length? (unless noted otherwise)
% NOTE: P.quant has a significant effect (as small AM can be covered up by splatter)
% --- wf freqs (unless noted oterwise, noise is unique to a given peak or pair)

P.f1= 802; P.f2= 1130; P.f3= 1350; % p1-p3 = narrower FM pair (p1&2) w/ unique AM and FM atop single wide peak (p3)
P.f4= 2134; P.f5= 2467; % p4&p5 = small sinusoids, p4 (AM) noisy, the other not
P.f6= 3056;  % p6 = small "wide" sinusoid via FM and with AM
P.f7= 3556;  % p7 = smaller "wide" sinusoid via FM and with AM
P.f8= 4235;  % p8 = try to get an Anolis-like peak (i.e., w/ AM and FM timescales similar to data)
P.f9= 4925; P.f10= 5530; P.f11= 5290; % p17-p19 = narrower FM pair (p9&10) w/ unique AM and FM atop single wide peak (p11)
% ==============================================
% ===== bookeeping
In.N= In.length*In.SR;  % total # of points in wavefrom(s)
freqL= [0:In.N/2]; freqL= In.SR*freqL./In.N; % freq. array for ENTIRE waveform
freqS= [0:In.L/2]; freqS= In.SR*freqS./In.L; % freq. array for AVERAGED segments
%NsA= round(P.delayA*0.001*In.SR); % sample # for start for X delay

% ===== create several waveforms to act as "peaks"

% -- wf1-3 = one sinusoid w/ no Brownian noise, no additive noise,
% (unique, Brownian-shaped) phase noise
P.N= In.N; P.f= P.f1; P.A= 0.1; P.bF= 0; P.bA= 0.2; P.pFB= 0; P.pNB= 1; P.pNBa= 0.55; P.pNr= 0.0008;
tempA= funcNoisySin(P); wf1= tempA(1:In.N,2);
P.f= P.f2; P.A= 0.3; tempA= funcNoisySin(P); wf2= tempA(1:In.N,2);
P.f= P.f3; P.A=0.1; P.bA= 0.2; P.pNr= 0.004; tempA= funcNoisySin(P); wf3= tempA(1:In.N,2);
% -- wf4 = lone small noisy (20% Brownian + 5% additive) sinusoid @ P.f1
P.N= In.N; P.f= P.f4; P.A=0.005; P.nA= 0; P.pNB= 0;
tempA= funcNoisySin(P); wf4= tempA(:,2);
% -- wf5 = lone small noiseless sinusoid @ P.f2
P.nA= 0; P.bA= 0; P.f= P.f5; P.A=0.005;
tempB= funcNoisySin(P); wf5= tempB(:,2);
% -- wf6 = lone small noisy "wide" sinusoid
P.nA= 0; P.bA= 0; P.f= P.f6; P.A=0.008; P.pNB= 1; P.pNBa= 0.55; P.pNr= 0.0008;
tempB= funcNoisySin(P); wf6= tempB(:,2);
% -- wf7 = lone smaller noisy "wide" sinusoid
P.nA= 0; P.bA= 0.3; P.f= P.f7; P.A=0.003; P.pNB= 1; P.pNBa= 0.55; P.pNr= 0.001;
tempB= funcNoisySin(P); wf7= tempB(:,2);

% -- wf8 = lone peak with AM and FM scales similar to Anolis 
P.f= P.f8; P.A=0.02; P.nS= 0; P.nA= 0.5;
P.bA= 10.0; P.bSign= 0; P.bNr= 0.008;
P.pNB= 1; P.pNBa= 0.6; P.pNr= 0.002;
P.pNI= 0; P.pNa= 0.1;
tempB= funcNoisySin(P); wf8= tempB(:,2);

% -- wf9-11 = sinusoid pair (p9&10) w/ (same) Brownian noise, no additive noise,
% and (same Brownian-shaped) phase noise, all set atop a
% very wide peak (p11) with its own unique AM and FM
P.N= In.N; P.f= P.f9; 
P.bF= 1; P.bA= 0.2; 
P.pFB= 1; P.pNB= 1; P.pNBa= 0.35; P.pNr= 0.001;
tempA= funcNoisySin(P); wf9= tempA(1:In.N,2);
P.f= P.f10; tempA= funcNoisySin(P); wf10= tempA(1:In.N,2);
P.f= P.f11; P.bA= 0.2; 
P.pNBa= 0.6; P.pNr= 0.005; tempA= funcNoisySin(P); wf11= tempA(1:In.N,2);


% =====
% --- compute analytic signal for two specified waveforms 
sig1= wf9;  sig2=wf10;
AS1= hilbert(sig1);   AS2= hilbert(sig2);  
% --- extract relevant waveforms for correlation analysis
env1= abs(AS1); env2= abs(AS2);    
ang1= (angle(AS1)); ang2= (angle(AS2));  % inst phase
IF1= gradient(unwrap(ang1),1/In.SR)/(2*pi);  % inst freq
IF2= gradient(unwrap(ang2),1/In.SR)/(2*pi);
t=linspace(0,numel(env1)/In.SR,numel(env1));    % create array of time values
% ===== add to get a summed waveform
wf= wf1+wf2+wf3+wf4+wf5+wf6+wf7+wf8+wf9+wf10+wf11;
% ===== spectral related calcs.
wfTspec= rfft(wf);   % compute spectrum
NN= floor(In.N/In.L);  % total # of possible segments for spec averaging
% -- do spec. averaging
for nn=1:NN
    indx= (nn-1)*In.L+1; % determine "next" index
    tempT= wf(indx:indx+In.L); % grab segment
    tempS(nn,:)= abs(rfft(tempT)); % compute spec. and mag of such
end
avgS= mean(tempS,1);  % compute (spectrally-)avgd. mags.

% ===== Fig.1: Plot (entire) ENV and IF for AS1 and AS2 
if 1==1
    figure(1); clf; set(gcf,'color','w');
    subplot(211)
    hAS1= plot(t,env1,'b','LineWidth',2); hold on; grid on;
    hAS2= plot(t,env2,'r','LineWidth',2);
    xlabel('Time [s]'); ylabel('Env. of analytic signal');
    legend([hAS1 hAS2],'AS1','AS2');
    subplot(212)
    hAS1= plot(t,IF1/1000,'b','LineWidth',2); hold on; grid on;
    hAS2= plot(t,IF2/1000,'r','LineWidth',2);
    xlabel('Time [s]'); ylabel('Inst. Freq [kHz]'); ylim([0.5 10]); 
end

% ===== Fig.2: Plot short segments for sig1 and sig2
if 1==1
    figure(2); clf; set(gcf,'color','w');
    plot(t(indx:indx+In.L),sig1(indx:indx+In.L),'b','LineWidth',2); hold on; grid on;
    plot(t(indx:indx+In.L),sig2(indx:indx+In.L),'r','LineWidth',1);
    legend('sig1','sig2'); title('Short (end) snippets of specified sig1 and sig2');
end

% ===== Fig.3: Plot mag. of spectrum: top=entire, bottom= spec. averaged
if 1==1
    figure(3); clf; set(gcf,'color','w');
    subplot(211)
    plot(freqL/1000,db(wfTspec),'b.-','LineWidth',1); hold on; grid on;
    xlabel('Frequency [kHz]'); ylabel('Amplitude [dB]');
    title('Spectrum of entire waveform'); xlim([0.5 10]);
    subplot(212)
    plot(freqS/1000,db(avgS),'b.-','LineWidth',1); hold on; grid on;
    xlabel('Frequency [kHz]'); ylabel('Amplitude [dB]'); xlim([0.5 10]);
    %title('Spectrally-averaged Spectrum');     
end


% --- Key params to funcNoisySin.m {default vals if unspecified}
% >>> Base sinusoid
% o P.N --> # of points for window {16384}
% o P.SR --> sample rate [Hz] {44100}
% o P.f -->  freq. of sinusoid [Hz] {2000} NOTE: freq. is quantized by default
% o P.phi -->  phase offset (should be between 0 and 2*pi) {0}
% o P.A --> sinusoid amplitude {1}
% >>> Additive noise
% o P.nA --> percent. (re P.A) ampl. of additive noise {0.05}
% o P.nF --> boolean to freeze additive noise {0}
% o P.nSeed --> seed ID for additive noise (reqs. P.nF=1)
% >>> Brownian noise (re envelope)
% o P.bNr --> percent. (re P.N, i.e., time-wise) for Brownian noise envelope {1/500}
% o P.bA --> percent. (re P.A) ampl. Browninan noise {0.2?}
% o P.bF --> boolean to freeze Brownian noise {0}
% o P.bSeed --> seed ID for Browninan noise (reqs. P.nF=1)
% o P.bSign --> boolean to "unsign" Brownian envelope {0}
% >>> Phase noise
% o P.pN --> boolean to add inst. phase noise {0}
% o P.pNa --> degree of inst. phase noise {0.1?} (reqs. P.phaseN=1)
% o P.pNr --> percent. (re P.N) for Brownian phase noise env {0.0005} (reqs. P.phaseN=1)
