clc;
clear;
close all;
%读入10%-46%co气体的数据
CO(1,:)=load('CO5.txt');
CO(2,:)=load('CO6.txt');
CO(3,:)=load('CO7.txt');
CO(4,:)=load('CO8.txt');
CO(5,:)=load('CO9.txt');
CO(6,:)=load('CO10.txt');
CO(7,:)=load('CO11.txt');
CO(8,:)=load('CO12.txt');
CO(9,:)=load('CO13.txt');
CO(10,:)=load('CO14.txt');
CO(11,:)=load('CO15.txt');
CO(12,:)=load('CO16.txt');
CO(13,:)=load('CO17.txt');
CO(14,:)=load('CO18.txt');
CO(15,:)=load('CO19.txt');
CO(16,:)=load('CO20.txt');
CO(17,:)=load('CO21.txt');
CO(18,:)=load('CO22.txt');
CO(19,:)=load('CO23.txt');
[m,n]=size(CO(1,:));
%横坐标波束
for clc=1:1:n
CO_X(1,clc)=9501-clc*(4000/n);%x = -pi:pi/10:pi;
end
%截取参数jie1,jie2
jie1=1;
jie2=2075;
figure(1);
plot(CO_X(1,jie1:jie2),CO(1,jie1:jie2),'-b');
hold on;
plot(CO_X(1,jie1:jie2),CO(2,jie1:jie2),'-g');
hold on;
plot(CO_X(1,jie1:jie2),CO(3,jie1:jie2),'-g');
hold on;
plot(CO_X(1,jie1:jie2),CO(4,jie1:jie2),'-g');
hold on;
plot(CO_X(1,jie1:jie2),CO(5,jie1:jie2),'-g');
hold on;
plot(CO_X(1,jie1:jie2),CO(6,jie1:jie2),'-g');
hold on;
plot(CO_X(1,jie1:jie2),CO(7,jie1:jie2),'-g');
hold on;
plot(CO_X(1,jie1:jie2),CO(8,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(9,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(10,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(11,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(12,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(13,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(14,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(15,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(16,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(17,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(18,jie1:jie2),'-r');
hold on;
plot(CO_X(1,jie1:jie2),CO(19,jie1:jie2),'-r');
title('NaCl');
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解

%取2000到2300预处理
COyu=CO(:,jie1:jie2);
COyu_x=CO_X(1,jie1:jie2);
[COyu_m, COyu_n]=size(COyu);
co_mean=mean(COyu);
COyu_y=([10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46])';

%数据中心化(Mean centering)
[CO_MCX,CO_MX] = center(COyu);
figure(2);
%plot(COyu_x(1,:),CO_MCX(1,:),'-r');
for clc=1:1:COyu_m
plot(COyu_x(1,:),CO_MCX(clc,:),'-r');
hold on;
end
title('数据中心化(Mean centering)'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解

%标准化Autoscales
[ax,mx,stdx] = auto((COyu)');
ax=ax';
figure(3);
for clc=1:1:COyu_m
plot(COyu_x(1,:),ax(clc,:),'-r');
hold on;
end
title('标准化Autoscales'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解

%归一化Normalize
[COyu_Nor] = normaliz(COyu);
figure(4);
for clc=1:1:COyu_m
plot(COyu_x(1,:),COyu_Nor(clc,:),'-r');
hold on;
end
title('归一化Normalize'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解
% z=[1,2,3;3,4,5];
% [m,n]=size(z);
% nx=z;
% nm=zeros(m,1);
% for i = 1:m
% nm(i)=norm(nx(i,:));
% nx(i,:)=nx(i,:)/nm(i);
% end

%移动窗口平滑/Moving-average method???
COyu_nir=nirmaf(COyu,9);
figure(5);
for clc=1:1:COyu_m
plot(COyu_x(1,:),COyu_nir(clc,:),'-r');
hold on;
end
title('移动窗口平滑光谱'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解

%SavitZky一Golay卷积平滑法及求导
COyu_S_G =savgol(COyu,15);%光谱，窗口大小；多项式项数；一阶求导;平滑;
COyu_S_G1 = savgol(COyu,7,3,1);%一阶求导
COyu_S_G2 = savgol(COyu,7,3,2);%二阶求导
figure(6);
for clc=1:1:COyu_m
plot(COyu_x(1,:),COyu_S_G(clc,:),'-r');
hold on;
end
title('SavitZky-Golay卷积平滑法'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解
figure(7);
for clc=1:1:COyu_m
plot(COyu_x(1,:),COyu_S_G1(clc,:),'-r');
hold on;
end
title('SavitZky-Golay一阶求导'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解
figure(8);
for clc=1:1:COyu_m
plot(COyu_x(1,:),COyu_S_G2(clc,:),'-r');
hold on;
end
title('SavitZky-Golay二阶求导'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解
%title('SavitZky一Golay卷积平滑法及一、二阶求导'); 

%直接差分一、二阶求导
d1=diff((COyu)',1);%
d2=diff((COyu)',2);%
d1=d1';
d2=d2';
figure(9);
for clc=1:1:COyu_m
plot(COyu_x(1,2:1:end),d1(clc,:),'-r');
hold on;
end
title('直接差分一阶求导'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解
figure(10);
for clc=1:1:COyu_m
plot(COyu_x(1,3:1:end),d2(clc,:),'-r');
hold on;
end
title('直接差分二阶求导'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解

%多元散射校正(MSC)
COyu_msc = msc(COyu,co_mean);
figure(11);
for clc=1:1:COyu_m
plot(COyu_x(1,:),COyu_msc(clc,:),'-r');
hold on;
end
title('多元散射校正(MSC)'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解

%标准正态变量交换snv
COyu_snv = snv(COyu);
figure(12);
for clc=1:1:COyu_m
plot(COyu_x(1,:),COyu_snv(clc,:),'-r');
hold on;
end
title('标准正态变量交换snv'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%直接信号校正dosc
[Z,W,P,T] = dosc(COyu,COyu_y,2,1E-3);
figure(13);
for clc=1:1:COyu_m
plot(COyu_x,Z(clc,:),'-r');
hold on;
end
title('直接信号校正dosc'); 
xlabel('wavenumber/cm-1'); % x轴注解
ylabel('Absorbance Unit'); % y轴注解
legend('NaCl'); % 图形注解
PPPPP=Z-COyu;