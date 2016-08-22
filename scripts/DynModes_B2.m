clear

%
% Script calculates dynamic vertical modes for a mooring and fits the
% velocity profile to these modes. 
% 
% Matt Rayson
% July 2008
% UWA
%

%--------------------------------------------------------------------------
load('O:\Browse Field Data\ModelValidationScripts\B2_data.mat');
outfile='O:\Browse Field Data\ModelValidationScripts\B2_modaluv.mat';
nmodes=6; % (Includes barotropic mode) set to zero to use number of current meters
filton=1; %high pass filter to remove low frequency currents
shelfang=39; % angle of shelf with +ve east
numpts=180;
%--------------------------------------------------------------------------

if nmodes==0
    nmodes=length(data.veldepths);
end

depths_B2=data.totaldepth+data.tempdepths;
z_i=data.veldepths+data.totaldepth;
avgtemp_B2=data.temp;

[numpoints,A]=size(avgtemp_B2);
% low-pass filtering density
temp_temp=avgtemp_B2; clear avgtemp_B2
lowrate = 1/(30*30); % 30 hours
% samplerate = 2/60; % 1 minute
Wn = [lowrate];
[C,D]= butter(5,Wn,'low');
avgtemp_B2=zeros(size(temp_temp));
for n = 1:A
    avgtemp_B2(:,n)= filtfilt(C,D,temp_temp(:,n)); % lowpass filter values
end

% computing buoyancy frequency
S = 34.6*ones(size(avgtemp_B2)); %assumed salinity
dep = depths_B2;
LAT = data.lat*ones(size(dep));
% calculating pressure using CSIRO seawater toolbox
pres = sw_pres(-dep,LAT);
P = zeros(size(avgtemp_B2));
for n = 1:numpoints
    P(n,:) = pres(:);
end
dens_B2 = sw_dens(S,avgtemp_B2,P); % low frequency density for buoyancy calc
dens = sw_dens(S,data.temp,P); % actual density for energy calcs

%calculating buoyancy frequency
rho0=mean(mean(dens_B2(~isnan(dens_B2)))); %mean density 
N_B2 = zeros(numpoints,A);
drho_dz = zeros(numpoints,A);
mid_rho = zeros(numpoints,A);
% drho_dz should be negative which means depth interval needs to be
% positive when going from top to bottom
for zz = 1:A-1
        depth_int=(dep(zz)-dep(zz+1));
        drho_dz(:,zz)=(dens_B2(:,zz)-dens_B2(:,zz+1))./(depth_int);
        mid_rho(:,zz)=(dens_B2(:,zz)+dens_B2(:,zz+1))/2;
end
drho_dz(:,A)=drho_dz(:,A-1);
mid_rho(:,A)=mid_rho(:,A-1);
drho_dz(drho_dz>0)=0; % removing regions of instability to calculate BVF
for zz = 1:A
     N_B2(:,zz)=sqrt(-9.81./mid_rho(:,zz).*drho_dz(:,zz));
end

% N_B2=2*pi./N_B2./60;
Nsq=[N_B2(:,1).^2,N_B2(:,:).^2]; %adds surface value for plotting
p=[0;pres'];

ntime=length(data.time);
% ntime=1000;

if filton==1 
    highrate = 1/(30*3600); % 30 hours
    samplerate = 1/60; % 1 minute
    Wn = [highrate/samplerate];
    [C,D]= BUTTER(5,Wn,'high');
    utemp= filtfilt(C,D,data.u); % highpass filter values
    vtemp= filtfilt(C,D,data.v);
    clear data.u data.v
    lowrate = 1/(3*3600); % 3 hours
    Wn = [lowrate/samplerate];
    [C,D]= BUTTER(5,Wn,'low');
    data.u=filtfilt(C,D,utemp);
    data.v=filtfilt(C,D,vtemp);
%     data.u=utemp;
%     data.v=vtemp;
end
% rotate velocity vectors to align with shelf
% rotate baroclinic velocities to align with alongshelf cross-shelf
% Rotation Matrix 
%
%         R = [ cos(ang) sin(ang)][u] = [u']
%             [-sin(ang) cos(ang)][v]   [v']
ang=deg2rad(shelfang);
%u (cross shelf)
U=data.u(:,:).*cos(ang)+data.v(:,:).*sin(ang);
V= data.v(:,:).*cos(ang)- data.u(:,:).*sin(ang); 
data.u=U;clear U
data.v=V;clear V

u=zeros(ntime,length(z_i),nmodes-1);
ubar=zeros(ntime,length(z_i));
v=zeros(ntime,length(z_i),nmodes-1);
vbar=zeros(ntime,length(z_i));

E_raw=zeros(ntime,1);
E_modes=zeros(ntime,nmodes);

% x=zeros(length(data.veldepths),1);
% xproj=max(axlims)*ones(length(data.veldepths),1);
% yproj=max(axlims)*ones(length(data.veldepths),1);
% y=zeros(length(data.veldepths),1);
% w=zeros(length(data.veldepths),1);
% z_t=data.tempdepths+data.totaldepth;
% err=zeros(ntime,2);
% contourvals=[16:2:30];
[ssh,ConList]=tide_pred('Model_tpxo7.1',data.time(1:60:end,1)-8/24,data.lat,data.lon,'z');

shelfang=-39; % angle of shelf with +ve east
ctr=numpts;
wh=waitbar(0,'Modal decomposition of velocity...');
for t=1:ntime % do as for loop 
    waitbar(t/ntime,wh);
    
    a_u=data.u(t,:); % do u and v separately
    a_v=data.v(t,:); % do u and v separately
    
    %computing eigenmodes for each timestep
    if ctr==numpts
         N=Nsq(t,:)';
        [wmodes,pmodes,ce]=dynmodes(N,p,nmodes);
        ctr=1;
    else 
        ctr=ctr+1;
    end
    
    z=[0;depths_B2(1:end)'];

    % finding the vertical modes at the current meter depths
    if nmodes>length(z_i); %number of modes to include
        error(['Number of modes must be <= ',num2str(length(z_i))])
    end
    F_n=zeros(length(z_i),nmodes); % barotropic and first three baroclinic modes

    % defining the barotropic mode as unity
    % (try using a log-profile later)
    F_n(:,1)=1;

    %interpolating baroclinic modes derived from dynmodes.m
    
    for n =1:nmodes-1
        F_n(:,n+1)=interp1(z,pmodes(:,n),z_i','linear');
        %normalizing horizontal modes (not really necessary)
        F_n(:,n+1)=F_n(:,n+1)/max(abs(F_n(:,n+1)));
    end
    
    [An_u ,STDX,MSE]=lscov(F_n,a_u');
     err(t,1)=MSE;
    ubar(t,:)=F_n(:,1).*An_u(1,1);
    for n =1:nmodes-1       
        u(t,:,n)=F_n(:,n+1).*An_u(n+1,1);
    end
    [An_v ,STDX,MSE]=lscov(F_n,a_v');
     err(t,2)=MSE;
    vbar(t,:)=F_n(:,1).*An_v(1,1);
   
    for n =1:nmodes-1        
        v(t,:,n)=F_n(:,n+1).*An_v(n+1,1);
    end
    %Energy Calculations
    dens_uv=interp1(depths_B2,dens(t,:),z_i,'linear');
    E_raw(t,1)=trapz(abs(z_i),0.5*(a_u.^2+a_v.^2));
    E_modes(t,1)=trapz(abs(z_i),0.5*(ubar(t,:).^2+vbar(t,:).^2));
    for n = 1:nmodes-1
        E_modes(t,n+1)=trapz(abs(z_i),0.5*(u(t,:,n).^2+v(t,:,n).^2));
    end
    
end
close(wh);
% Potential Energy Calculation
PE=zeros(size(dens));
depths=data.tempdepths;
for n = 1:length(depths)
    PE(:,n)=dens(:,n)*9.81*depths(n);
end  
sum_PE=trapz(depths,PE,2);

% Saving data to a structure
modedata.veldepths=data.veldepths;
modedata.tempdepths=data.tempdepths;
modedata.totaldepth=data.totaldepth;
modedata.lat=data.lat;
modedata.lon=data.lon;
modedata.time=data.time;
modedata.rhomean=dens_B2;
modedata.rho=dens;
modedata.Nsq=Nsq;
modedata.u=u;
modedata.v=v;
modedata.uraw=data.u;
modedata.vraw=data.v;
modedata.ubar=ubar;
modedata.vbar=vbar;
modedata.ssh=ssh; %derived from TPXO
modedata.E=E_raw;
modedata.Emodes=E_modes;
modedata.PE=sum_PE;

save(outfile,'modedata');

    
figure
plot(E_raw,'k');hold on
plot(sum(E_modes,2),'r')
ylabel('KE (m^3/s^-^2)')
legend('RAW KE','Sum Modal KE')

figure
plot(E_raw,'k');hold on
plot(E_modes)
ylabel('KE (m^3/s^-^2)')
legend('Total','Barotropic','Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Mode 7')

perc_E2=zeros(size(E_modes));
for n = 1:nmodes
    perc_E2(:,n)=E_modes(:,n)./sum(E_modes,2)*100;
end
Efilt=movavgfilt(perc_E2,7*24*2*30);
figure
plot(Efilt)
ylabel('% Energy')
legend('Barotropic','Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Mode 7')

perc_E=sum(E_modes)/sum(sum(E_modes))*100
