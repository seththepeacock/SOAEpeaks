function [Data] = funcNoisySin(P);
% *** funcNoisySIN.m ***       2018.03.07  (updated 2023.06.13)     C. Bergevin

% Function to make a noisy sinusoidal waveform (e.g., to be used for
% testing inter-waveform correlation analysis codes. This sort of waveform
% can serve as "ground truth" (e.g., xcorr analysis codes). 
% Allows for the following:
% 1. additive noise (via P.nA)
% 2. Brownian noise envelope for AM (via P.bA)
% 3. Brownian phase noise for FM (via P.pN)
% 4. inst. noise for either the above
% Noise can also be independently be "frozen". 
% ---
% Input: structure P w/ params as specified below (can be left empty and
% default values will be used)
% ---
% Output: P.Nx3 array, 1st column is time [s], 2nd the noisy signal [arb],
% 3rd the Brownian envelope of ampl. [arb]
% Note: If P.pN= 1, output is P.Nx4, where 4th column is Brownian envelope of
% phase noise (re P.f)
% ---
% Usage (e.g.,): >> x=funcNoisySin; plot(x(:,1),x(:,2));
% Ex.2 > P.N= In.N; P.f= P.f1;
%      > tempA= funcNoisySin(P); wf1= tempA(:,2);
% ---
% Notes (see bottom of code for additional bits)
% o According to Middleton (1960; An Introduction to Statistical
% Communication Theory), FM is a bit tricky (i.e., distinction between 
% phase and frequency modulation). The noise term re the
% frequency appears as an integral, which here means the noise is basically
% being included as a derivative. In terms of coding, if phi(t) is the noise term:
% * (not correct, but seems so) sin(2*pi*t*(fQ+phi));
% * (way implemented) sin(2*pi*(t*fQ + phi));
% which is basically equivalent to sin(2*pi*t*(fQ+phi/t)) (buts avoids the
% issue of a NaN for the first value where t=0).
% --> Ultimately, the FM created by phi(t) goes as -d(phi)/dt
% o ref. code for Brownian noise is EXhoRK4noisy.m
% o Brownian noise envelope is (multiplicatively) applied after the
% additive is introduced
% o By default, Brownian envelope is signed (i.e., there can be phase
% reversals). These can be removed by setting P.bSign= 1, which also adds a
% ~10% vert. offset to get rid of cusps (i.e., places where the signal
% would go to zero in a differentially unsmooth way)
% o re freezing the noise: nixed legacy code  > randn('state',P.nSeed);
% and replaced w/ > rng(P.nSeed);

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
% o P.bSeed --> seed ID for Browninan noise (reqs. P.bF=1)
% o P.bSign --> boolean to "unsign" Brownian envelope {0}
% >>> Phase noise
% NOTE: Totally rewrote, so I need to update this list directly below
% o P.pN --> boolean to add inst. phase noise {0}
% o P.pNa --> degree of inst. phase noise {0.1?} (reqs. P.phaseN=1)
% o P.pNr --> percent. (re P.N) for Brownian phase noise env {0.0005} (reqs. P.phaseN=1)
% o P.pF --> boolean to freeze phase noise {0}
% o P.pSeed --> seed ID for Browninan noise (reqs. P.pF=1)

% % ===============================================================================
%if ~(exist('P')),  clear; P=[];   end         % uncomment to run as a script
if (nargin < 1 | isempty (P)), P = [];           end;  % uncomment to run as a function
% ---
if (~isfield(P,'N')), P.N= 16384; end  % # of points for window {16384}
if (~isfield(P,'SR')), P.SR= 44100; end  % sample rate [Hz] {44100}
if (~isfield(P,'f')), P.f= 2000; end  % freq. of sinusoid [Hz] {2000}
if (~isfield(P,'phi')), P.phi= 0; end  % phase offset (should be between 0 and 2*pi) {0}
if (~isfield(P,'A')), P.A= 1; end  % sinusoid amplitude {1}
% --- noise params.
% >>> additive noise
if (~isfield(P,'nS')), P.nS= 1; end  % boolean to add additive noise {1} (if 0, P.nA will be set to 0)
if (~isfield(P,'nA')), P.nA= 0.05; end  % additive noise amplitude {0.05}
if (~isfield(P,'nF')), P.nF= 0; end  % boolean to freeze additive noise {0}
if (~isfield(P,'nSeed')), P.nSeed= 114; end  % seed ID for additive noise (reqs. P.nF=1)
% >>> Brownian noise (re envelope)
if (~isfield(P,'bS')), P.bS= 1; end  % boolean to add Brownian noise (re envelope) {1} (if 0, P.bA will be set to 0)
if (~isfield(P,'bNr')), P.bNr= 0.002; end  % percent. (re P.N) for Brownian noise envelope {1/500}
if (~isfield(P,'bA')), P.bA= 0.2; end  % Browninan noise amplitude {0.2?}
if (~isfield(P,'bF')), P.bF= 0; end  % boolean to freeze Brownian noise {0}
if (~isfield(P,'bSeed')), P.bSeed= 14; end  % seed ID for Browninan noise (reqs. P.bF=1)
if (~isfield(P,'bSign')), P.bSign= 1; end  % boolean to "unsign" Brownian envelope {0}
% >>> inst. phase noise
if (~isfield(P,'pNI')), P.pNI= 0; end  % boolean to add inst. phase noise [creates pNoiseI] {0}
if (~isfield(P,'pNa')), P.pNa= 0.01; end  % degree of inst. phase noise {0.1?} (reqs. P.pNI=1)
if (~isfield(P,'pF')), P.pFI= 0; end  % boolean to freeze inst. phase (Brownian) noise {0}
if (~isfield(P,'pSeed')), P.pISeed= 14; end  % seed ID for inst. phase noise (reqs. P.nF=1)
% >>> Brownian FM phase noise
if (~isfield(P,'pNB')), P.pNB= 0; end  % boolean to add Borwnian FM [creates pNoiseB] {0}
if (~isfield(P,'pNBa')), P.pNBa= 0.05; end  % amplitude for Brownian FM envelope {0.1?}  (reqs. P.pNB=1)
if (~isfield(P,'pNr')), P.pNr= 0.0005; end  % percent. (re P.N) for Brownian phase noise env {0.0005} 
if (~isfield(P,'pFB')), P.pFB= 0; end  % boolean to freeze phase (Brownian) noise {0}
if (~isfield(P,'pSeedB')), P.pSeedB= 14; end  % seed ID for Brownian FM phase noise (reqs. P.nF=1)
% --- spectral-related stuff not directly used (but potentially useful post-processing)
if (~isfield(P,'Npts')), P.Npts= 8192; end  % length of fft window (# of points) [should ideally be 2^N]
if (~isfield(P,'quant')), P.quant= 1; end  % boolean to "quantize" freq. re FFT window {1}
% % ===============================================================================
% --- quantize the freq. (so to have an integral # of cycles in time window)
df = P.SR/P.Npts;
if (P.quant==1), fQ= ceil(P.f/df)*df;   else fQ= P.f; end
% --- some derived quantities
dt= 1/P.SR;  % spacing of time steps
freq= [0:P.Npts/2];    % create a freq. array (for FFT bin labeling)
freq= P.SR*fQ./P.Npts;
t=[0:1/P.SR:(P.N-1)/P.SR];  % create an array of time points, Npoints long
P.bN= round(P.N*P.bNr);  % # of points re Browninan noise in ampl.
P.pNp= round(P.N*P.pNr);  % " (but for the phase noise)
if (P.pNp==1),  P.pNp= P.pNp+1; end  % make sure there are at least two pts for interp.
% --- additive noise
if (P.nS==0), P.nA= 0; end
if (P.nF==1),   rng(P.nSeed);  end
aNoise= P.A* P.nA* randn(P.N,1)';  % note that P.nA is a percent. re P.A
if (P.nF==1),   rng shuffle;  end   % reset random # gen.
% --- Brownian noise (via spline interp. of additional noise)
if (P.bS==0), P.bA= 0; end
if (P.bF==1),   rng(P.bSeed,'twister');  end
baseN= randn(P.bN,1);  % create baseline low-pass noise
if (P.bF==1),   rng shuffle;  end   % reset random # gen
tN= linspace(0,max(t),P.bN);          % time array for discrete noise points (re ampl)
bNoise= P.A* P.bA* spline(tN,baseN,t);  % spline interp.; note that P.bA is a percent. re P.A
%if (P.bA==0),   bNoise= ones(1,numel(t)); end  % kludge to turn Brownian envelop. noise off
if (P.bSign==1),  bNoise= abs(bNoise)+0.1*max(bNoise);  end   % "unsign" 
% --- phase noise I: low-pass Brownian FM noise (pNoiseB)
if (P.pNB==1),  if (P.pFB==1), rng(P.pSeedB);  end
    baseNp= randn(P.pNp,1);  % create baseline
    tNp= linspace(0,max(t),P.pNp);          % time array for discrete noise points
    pNoiseB= 1+ (P.pNBa* spline(tNp,baseNp,t));    % spline interpolation
    if (P.pFB==1),  rng shuffle;  end   % reset random # gen
else pNoiseB= ones(1,P.N);
end
% --- phase noise II: inst. phase noise (pNoiseI)
if (P.pNI==1),  if (P.pFI==1), rng(P.pSeedI);  end;
    pNoiseI= P.pNa* randn(1,P.N);
    if (P.pFI==1),   rng shuffle;  end;
else pNoiseI= zeros(1,P.N);
end
%figure(77); plot(t,fQ+(pNoiseB+ pNoiseI));
% ================
% ---  base sinusoid (w/ phase noise)
base= P.A*sin(2*pi*(t.*fQ+(pNoiseB+ pNoiseI)+P.phi));
% --- now factor in envelope modulation (AM) and additive noise
%wf= bNoise.*(base+ aNoise);
wf= (1+bNoise).*(base+ aNoise);
if (P.pNB==1),   Data= [t' wf' bNoise' pNoiseB'];
else    Data= [t' wf' bNoise']; end
return
% ====================
% Various Notes
% o (2019.11.09) Updating "phase noise"; added line re if P.pNp= P.pNp+1 
% so there are at least two points in envelope; added boolenas to turn
% on/off additive and Brownian envelope noise
% o updating (2019.05.17) to allow for inst. phase to also be "noisy"
% (which I believe amounts to freq.-modulation). If you want "slow" Brownian-like 
% noise, set P.pNr= 0.0005 or the like. For something "faster", set P.pNr
% closer to unity. Can also plot output of that like plot(x(:,1),x(:,4));
% o ref. code for Brownian noise is EXhoRK4noisy.m
% o Brownian noise envelope is (multiplicatively) applied after the
% additive is introduced
% o By default, Brownian envelope is signed (i.e., there can be phase
% reversals). These can be removed by setting P.bSign= 1, which also adds a
% ~10% vert. offset to get rid of cusps (i.e., places where the signal
% would go to zero in a differentially unsmooth way)
% o re freezing the noise: nixed legacy code  > randn('state',P.nSeed);
% and replaced w/ > rng(P.nSeed);
% --- 2023.06.13 NOTES
% o Phase noise was not being correctly handled --> think this is now
% partially corrected.... Also fixed some other bugs (e.g., boolean to
% freeze the brownian phase noise)

% ====================
% Older legacy code bits (for ref if needed)
% ---  base sinusoid (w/ phase noise if specified)
% if (P.pNI==1),  base= P.A*sin(2*pi*fQ*t- P.phi+ pNoiseI);
%     %base= P.A*sin(2*pi*fQ*t.*(1+pNoise)- P.phi);
%     %base= P.A*sin(2*pi*t.*(fQ+pNoise)- P.phi);
%     %base= P.A*sin(2*pi*fQ*t.*pNoiseB- P.phi+ pNoiseI);
% elseif (P.pN==1),
% else    base= P.A*sin(2*pi*fQ*t-P.phi);  end
% -- older incorrect vers where IF fluct. increased w/ time
%base= P.A*sin(2*pi*fQ*t.*pNoiseB- P.phi+ pNoiseI);
% -- updated (kludge?) version
%base= P.A*sin(2*pi*t.*(fQ+(pNoiseB+ pNoiseI)./t)+P.phi);