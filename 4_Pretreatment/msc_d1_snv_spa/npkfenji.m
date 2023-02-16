clear all;
clc

xx = xlsread("D:\松针\songzhen\x.xls");
ck = xlsread("C:\Users\Administrator\Desktop\songzhen\else\ck.xlsx");
k1 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\k1.xlsx");
k2 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\k2.xlsx");
k3 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\k3.xlsx");
n1 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\n1.xlsx");
n2 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\n2.xlsx");
n3 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\n3.xlsx");
p1 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\p1.xlsx");
p2 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\p2.xlsx");
p3 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\p3.xlsx");

b = [ck;n1;n2;n3];

x = xx(1:176);
x=x';
x2 = x(1:175);
X = b(:,1:176);
b1 = b(:,1:176);
% 归一化
A1=mapminmax(b1,0,1);
% 一阶导
A2=diff(b1');A2=A2';
% MSC
bb=mean(b1);
A3=msc(b1,bb);
LinAlpha=0.25;           
ColorValue=[0.8500 0.3250 0.0980];

%1、氮素
figure 
% subplot(2,2,1)
% plot(x,b1(1:27,:),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x,b1(28:50,:),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x,b1(51:77,:),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x,b1(78:98,:),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;

plot(x,mean(b1(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
plot(x,mean(b1(28:50,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
plot(x,mean(b1(51:77,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
plot(x,mean(b1(78:98,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
title({'\fontname{Times New Roman}Origin'},'FontSize',12);
xlim([400, 1010]); 
set(gca,'FontSize',12,'Fontname', 'Times New Roman');
set(gca, 'Box', 'off');
legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');

% subplot(2,2,2)
% plot(x,mean(A1(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x,mean(A1(28:50,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x,mean(A1(51:77,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x,mean(A1(78:98,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}Mapminmax'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');
% 
% subplot(2,2,3)
% plot(x,mean(A3(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x,mean(A3(28:50,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x,mean(A3(51:77,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x,mean(A3(78:98,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}MSC'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');
% 
% 
% subplot(2,2,4)
% plot(x2,mean(A2(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x2,mean(A2(28:50,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x2,mean(A2(51:77,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x2,mean(A2(78:98,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}D1'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');


%2、磷素
% figure 
% subplot(2,2,1)
% plot(x,mean(b1(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x,mean(b1(28:54,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x,mean(b1(55:84,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x,mean(b1(85:113,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}Origin'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');
% 
% subplot(2,2,2)
% plot(x,mean(A1(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x,mean(A1(28:54,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x,mean(A1(55:84,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x,mean(A1(85:113,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}Mapminmax'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');
% 
% subplot(2,2,3)
% plot(x,mean(A3(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x,mean(A3(28:54,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x,mean(A3(55:84,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x,mean(A3(85:113,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}MSC'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');
% 
% 
% subplot(2,2,4)
% plot(x2,mean(A2(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x2,mean(A2(28:54,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x2,mean(A2(55:84,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x2,mean(A2(85:113,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}D1'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');
% 

%3、钾素
% figure 
% subplot(2,2,1)
% plot(x,mean(b1(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x,mean(b1(28:54,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x,mean(b1(55:80,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x,mean(b1(81:106,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}Origin'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');
% 
% subplot(2,2,2)
% plot(x,mean(A1(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x,mean(A1(28:54,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x,mean(A1(55:80,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x,mean(A1(81:106,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}Mapminmax'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');
% 
% subplot(2,2,3)
% plot(x,mean(A3(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x,mean(A3(28:54,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x,mean(A3(55:80,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x,mean(A3(81:106,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}MSC'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');
% 
% 
% subplot(2,2,4)
% plot(x2,mean(A2(1:27,:)),'-','LineWidth',1,'color',[0 0.4470 0.7410]);hold on;
% plot(x2,mean(A2(28:54,:)),'-','LineWidth',1,'color',[0.8500 0.3250 0.0980]);hold on;
% plot(x2,mean(A2(55:80,:)),'-','LineWidth',1,'color',[0.9290 0.6940 0.1250]);hold on;
% plot(x2,mean(A2(81:106,:)),'-','LineWidth',1,'color',[0.4940 0.1840 0.5560]);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength/nm'},'FontSize',12);
% ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}D1'},'FontSize',12);
% xlim([400, 1010]); 
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
% set(gca, 'Box', 'off');
% legend('\fontname{Times New Roman}CK','\fontname{Times New Roman}N1','\fontname{Times New Roman}N2','\fontname{Times New Roman}N3','Location','Northwest');

