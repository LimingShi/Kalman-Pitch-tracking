%
clc;clear all;close all;

[x,fs]=audioread('fa.wav');

SNR=10;
noise=sqrt(var(x)/10^(SNR/10))*randn(length(x),1);
noisy=x+noise;   
%% Initiate the algorithm
L=5;
f0=200;
a=amp_ls(x(1:300),2*pi*f0/fs*[1:L]);
x_ini=[2*pi*f0/fs;sqrt(sum(a.^2,2));0]; PHASE=atan(a(:,2)./a(:,1));
P=1e-6*eye(L+2,L+2); 
R_noise=var(noise);
Q= 1e-7*eye(L+1,L+1);Q(1,1)= (2*pi*0.2/fs)^2;
smooth_flag=1;


state_matrix=PitchEKF(noisy,L,x_ini,P,R_noise,Q,PHASE,smooth_flag);
plot([1:size(state_matrix,2)]/fs,state_matrix(1,:)/2/pi*fs);
xlabel('Time [s]');
ylabel('Frequency [Hz]');



function a=amp_ls(x,w);
%AMP_LS   amplitude estimator based on LS
x=x(:);
N=length(x);
Z=vandermonde_mads(w,N);
a=Z\x;
a=reshape(a,[],2);
end

function Z=vandermonde_mads(w,N);
w=w(:)';
Z=[cos(w.*[1:N]'),sin(w.*[1:N]')];
end


function [m_upS,P_upS]=PitchEKF(observation,L,init_state_value,P,...
    R_noise,Q,pha,smooth_flag)
% Extented kalman filter
% for pitch tracking; Observation is scalar form;
% For paper Instantaneous fundamental frequency estimation of non-stationary periodic
% signals using non-linear recursive filters 2014 in IET Signal Process.
% Liming Shi, 2016-11-9, AAU, Denmark

if size(observation,1)>size(observation,2)
    %     warning('The input matrix is transposed to make row as time axis');
    observation=observation';
end
F=eye(L+2);F(end,1)=1; %very important segment for producing the state transition matrix
C=[eye(L+1);zeros(1,L+1)];
G=toeplitz(zeros(L,1),[0;1;zeros(L+2-2,1)]);
B=zeros(L,L+2);B(:,1)=1:L;B(:,end)=1:L;




state=zeros(L+2,size(observation,2));
state(:,1)=init_state_value;
R=R_noise;                            % Observation noise covariance matrix
h=zeros(size(observation,1),1);        % Predicted observation value
P_OneStep=zeros(size(P,1),size(P,2),size(observation,2)+1);


%% Extented Kalman filtering for pitch tracking
for ii=2:size(observation,2)
    % one-step prediction of mean based on the previous estimation
    One_step_state=F*(state(:,ii-1));
    % state Covariance/Uncertainty from state transition equation
    % due to the nonlinearity of the state transition matrix. The F
    % should be the Jacobian matrix of the nonlinear state transformation
    % Compute the Jacobian matrix
    P_OneStep(:,:,ii)=F*P(:,:,ii-1)*F'+C*Q*C';
    % Jacobian matrix of the observation equaiton
    H=cos((B*One_step_state)'+pha')*G-(G*One_step_state)'*...
        diag(sin(B*One_step_state+pha))*(B);
    % Observation covariance from obervation equation
    O_covariance=(H*P_OneStep(:,:,ii)*H'+R);
    K=P_OneStep(:,:,ii)*H'*O_covariance^(-1); % Kalman gain,
    % Compute the one-step prediction residual
    h=(G*One_step_state)'*cos(B*One_step_state+pha);
    correction_factor=K*(observation(:,ii)-h);
    %plot(ii,(observation(:,ii-1)-h),'.');hold on; pause (1e-9);
    % state mean update using observation
    state(:,ii)= One_step_state+correction_factor;
    P(:,:,ii)=P_OneStep(:,:,ii)-K*H*P_OneStep(:,:,ii); % state covariance update
    
    
end
% Kalman smoother;
smooth=smooth_flag;
N=size(observation,2);
if(smooth)
    %These are the Rauch, Tung and Striebel recursions
    m_upS(:,N)     = state(:,N);
    P_upS(:,:,N)   = P(:,:,N);
    
    for k = (N-1):-1:1
        % Compute prediction steps for all but the last step
        sgain = (P(:,:,k)*F')/(F*P(:,:,k)*F' + C*Q*C');
        m_upS(:,k)     = state(:,k)  + sgain*(m_upS(:,k+1)  - F*(state(:,k)));
        P_upS(:,:,k)   = P(:,:,k)+ sgain*(P_upS(:,:,k+1) - P_OneStep(:,:,k+1))*sgain';
    end
else
    % Just package the filtered outputs
    m_upS = state;
    P_upS = P;
    PP_upS = [];
end
end





