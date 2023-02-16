clear all;
clc
ck = xlsread("C:\Users\Administrator\Desktop\songzhen\else\ck.xlsx");
n1 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\n1.xlsx");
n2 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\n2.xlsx");
n3 = xlsread("C:\Users\Administrator\Desktop\songzhen\else\n3.xlsx");

C = [ck;n1;n2;n3];
C1 = C(:,1:176);
A1 = mapminmax(C1,0,1);
A11 = mean(A1);
% % 标准化
% A2=zscore(b',0,1);A2=A2';
% % 一阶导
A3=diff(C1');A3=A3';
A31 = mean(A3);

% MSC
CCC=mean(C1);
A6=msc(C1,CCC);
A61 = mean(A6);

erty = 7;
erty2 = 16;

origin_cars = xlsread('C:\Users\Administrator\Desktop\songzhen\result\N\origin_cars特征波段.xls');
origin_spa = xlsread('C:\Users\Administrator\Desktop\songzhen\result\N\origin_spa特征波段.xls');
mapminmax_cars = xlsread('C:\Users\Administrator\Desktop\songzhen\result\N\mapminmax_cars特征波段.xls');
mapminmax_spa = xlsread('C:\Users\Administrator\Desktop\songzhen\result\N\mapminmax_spa特征波段.xls');
msc_cars = xlsread('C:\Users\Administrator\Desktop\songzhen\result\N\msc_cars特征波段.xls');
msc_spa = xlsread('C:\Users\Administrator\Desktop\songzhen\result\N\msc_spa特征波段.xls');
d1_cars = xlsread('C:\Users\Administrator\Desktop\songzhen\result\N\d1_cars特征波段.xls');
d1_spa = xlsread('C:\Users\Administrator\Desktop\songzhen\result\N\d1_spa特征波段.xls');

rrt = xlsread('C:\Users\Administrator\Desktop\songzhen\画图程序\guangputu.xls');
rrt1 = rrt(36:171,1)';
rrt2 = rrt(35:171,1)';
rrt3 = rrt(36:171,1)';
rrt4 = rrt(37:171,1)';


origin = xlsread("C:\Users\Administrator\Desktop\songzhen\origin.xls");
mapminmax = xlsread("C:\Users\Administrator\Desktop\songzhen\mapminmax.xls");
msc = xlsread("C:\Users\Administrator\Desktop\songzhen\msc.xls");
snv = xlsread("C:\Users\Administrator\Desktop\songzhen\snv.xls");
d1 = xlsread("C:\Users\Administrator\Desktop\songzhen\d1.xls");
xx = xlsread('C:\Users\Administrator\Desktop\songzhen\x.xls');

xxx = xx';
x = xxx(34:170);
x2 = xxx(35:170);
origin_1 = origin(:,34:170);
mapminmax_1 = mapminmax(:,34:170);
msc_1 = msc(:,34:170);
snv_1 = snv(:,34:170);
d1_1 = d1(:,34:169);

[coeff1,score1,latent1,tsquared1,explained1,mu1] = pca(origin_1,'NumComponents',5);
b1=coeff1(:,1:5);
[coeff2,score2,latent2,tsquared2,explained2,mu2] = pca(mapminmax_1,'NumComponents',5);
b2=coeff2(:,1:5);
[coeff3,score3,latent3,tsquared3,explained3,mu3] = pca(msc_1,'NumComponents',5);
b3=coeff3(:,1:5);
[coeff4,score4,latent4,tsquared4,explained4,mu4] = pca(d1_1,'NumComponents',5);
b4=coeff4(:,1:5);


str1 = {'\fontname{Arial}a)'};
str2 = {'\fontname{Arial}b)'};
str3 = {'\fontname{Arial}c)'};
str4 = {'\fontname{Arial}d)'};
str5 = {'\fontname{Arial}e)'};
str6 = {'\fontname{Arial}f)'};


% Selection feature wavelength by SPA
h = figure;
set(h,'position',[100 100 800 700]);
subplot(3,2,1);
plot(rrt2,mapminmax_spa(2,:),'k.-'),hold on;
% xlabel({'\fontname{Times New Roman}Wavelength(nm)'},'FontSize',12);
ylabel({'\fontname{Times New Roman}Frequency of occurrence'},'FontSize',12);
% title({'\fontname{Times New Roman}Mapminmax-SPA'},'FontSize',12);
xlim([500, 1010]); 
set(gca,'tickdir','out');
set(gca,'FontSize',12,'Fontname', 'Times New Roman');
set(gca,'ylim',[0 500]);
box off
ert = [6,13,43,50,54,64,75,79,120,130];
scatter(rrt2(ert),mapminmax_spa(2,ert),erty2,'k','filled');hold on;
text(rrt2(ert(1)-5),mapminmax_spa(2,ert(1))+30,num2str(rrt2(ert(1))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt2(ert(2)-5),mapminmax_spa(2,ert(2))+30,num2str(rrt2(ert(2))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt2(ert(3)-9),mapminmax_spa(2,ert(3))+30,num2str(rrt2(ert(3))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt2(ert(4)-5),mapminmax_spa(2,ert(4))+30,num2str(rrt2(ert(4))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt2(ert(5)-11),mapminmax_spa(2,ert(5))+30,num2str(rrt2(ert(5))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt2(ert(6)-10),mapminmax_spa(2,ert(6))-30,num2str(rrt2(ert(6))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt2(ert(7)-5),mapminmax_spa(2,ert(7))+30,num2str(rrt2(ert(7))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt2(ert(8)-5),mapminmax_spa(2,ert(8))+30,num2str(rrt2(ert(8))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt2(ert(9)-15),mapminmax_spa(2,ert(9))+30,num2str(rrt2(ert(9))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt2(ert(10)-5),mapminmax_spa(2,ert(10))+30,num2str(rrt2(ert(10))),'FontSize',12,'FontName','Times New Roman');hold on;
text(600,450,str1,'FontSize',15)




subplot(3,2,2);
qwert=mean(C1(1:27,35:171));
plot(x,qwert,'k-','LineWidth',1);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength(nm)'},'FontSize',12);
ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}Origin'},'FontSize',12);
xlim([500, 1010]); 
set(gca,'tickdir','out');
set(gca,'FontSize',12,'Fontname', 'Times New Roman');
set(gca, 'Box', 'off');
scatter(x(ert),qwert(ert),erty2,'k','filled');hold on;
for i=1:10
    bar(x(ert(i)),qwert(ert(i)),mapminmax_spa(2,ert(i))/500*10,'FaceColor',[0 0.4470 0.7410]);hold on;
end
text(600,0.55,str2,'FontSize',15)

subplot(3,2,3);
plot(rrt1,msc_cars(2,:),'k.-'),hold on;
% xlabel({'\fontname{Times New Roman}Wavelength(nm)'},'FontSize',12);
ylabel({'\fontname{Times New Roman}Frequency of occurrence'},'FontSize',12);
% title({'\fontname{Times New Roman}MSC-CARS'},'FontSize',12);
xlim([500, 1010]); 
set(gca,'tickdir','out');
set(gca,'FontSize',12,'Fontname', 'Times New Roman');
set(gca,'ylim',[0 500]);
box off
ert1 = [5,12,40,45,74,78,114,118,120,129];
scatter(rrt1(ert1),msc_cars(2,ert1),erty2,'k','filled');hold on;

text(rrt1(ert1(1)-4),msc_cars(2,ert1(1))+30,num2str(rrt1(ert1(1))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt1(ert1(2)-2),msc_cars(2,ert1(2))+30,num2str(rrt1(ert1(2))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt1(ert1(3)-15),msc_cars(2,ert1(3))+30,num2str(rrt1(ert1(3))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt1(ert1(4)-4),msc_cars(2,ert1(4))+30,num2str(rrt1(ert1(4))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt1(ert1(5)-15),msc_cars(2,ert1(5))+30,num2str(rrt1(ert1(5))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt1(ert1(6)),msc_cars(2,ert1(6))+30,num2str(rrt1(ert1(6))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt1(ert1(7)-15),msc_cars(2,ert1(7))+30,num2str(rrt1(ert1(7))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt1(ert1(8)-4),msc_cars(2,ert1(8))+30,num2str(rrt1(ert1(8))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt1(ert1(9)),msc_cars(2,ert1(9))+30,num2str(rrt1(ert1(9))),'FontSize',12,'FontName','Times New Roman');hold on;
text(rrt1(ert1(10)-4),msc_cars(2,ert1(10))+30,num2str(rrt1(ert1(10))),'FontSize',12,'FontName','Times New Roman');hold on;
text(600,450,str3,'FontSize',15)


subplot(3,2,4);
qwert=mean(C1(1:27,35:171));
plot(x,qwert,'k-','LineWidth',1);hold on;
% xlabel({'\fontname{Times New Roman}Wavelength(nm)'},'FontSize',12);
ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}Origin'},'FontSize',12);
xlim([500, 1010]); 
set(gca,'tickdir','out');
set(gca,'FontSize',12,'Fontname', 'Times New Roman');
set(gca, 'Box', 'off');
scatter(x(ert1),qwert(ert1),erty2,'k','filled');hold on;
for i=1:10
    bar(x(ert1(i)),qwert(ert1(i)),msc_cars(2,ert1(i))/500*10,'FaceColor',[0 0.4470 0.7410]);hold on;
end
text(600,0.55,str4,'FontSize',15)

subplot(3,2,5);
p1 = plot(x,b1(:,1),'LineWidth',1);
c1 = p1.Color;
p1.Color = [0 0.4470 0.7410];
hold on;
p2 = plot(x,b1(:,2),'g','LineWidth',1);
c2 = p2.Color;
p2.Color = [0.8500 0.3250 0.0980];
hold on;
p3 = plot(x,b1(:,3),'b','LineWidth',1);
c3 = p3.Color;
p3.Color = [0.9290 0.6940 0.1250];
hold on;
p4 = plot(x,b1(:,4),'k','LineWidth',1);
c4 = p4.Color;
p4.Color = [0.4940 0.1840 0.5560];
hold on;
% xlabel({'\fontname{Times New Roman}Wavelenth(nm)'},'FontSize',12);
ylabel({'\fontname{Times New Roman}Weight coefficient'},'FontSize',12);
% title({'\fontname{Times New Roman}Origin-PCA-weight'},'FontSize',12);
set(gca,'FontSize',12,'Fontname', 'Times New Roman');
xlim([500, 1010]); 
ylim([-0.2, 0.6]); 
ax = gca;
ax.XAxis.MinorTick = 'on';
ax.XAxis.MinorTickValues = 450:50:960;
ax.YAxis.MinorTick = 'on';
ax.YAxis.MinorTickValues = -0.3:0.05:0.6;
set(gca,'tickdir','out');
set(gca,'FontSize',12);
set(gca,'FontSize',12,'Fontname', 'Times New Roman');
set(gca, 'Box', 'off');

scatter(x(123),b1(123,1),erty2,'k','filled');hold on;
scatter(x(16),b1(16,2),erty2,'k','filled');hold on;
scatter(x(51),b1(51,2),erty2,'k','filled');hold on;
scatter(x(60),b1(60,2),erty2,'k','filled');hold on;
scatter(x(57),b1(57,3),erty2,'k','filled');hold on;
scatter(x(73),b1(73,3),erty2,'k','filled');hold on;
text(x(123-erty),b1(123,1)-0.04,num2str(x(123)),'FontSize',12,'FontName','Times New Roman');hold on;
text(x(16-erty),b1(16,2)+0.04,num2str(x(16)),'FontSize',12,'FontName','Times New Roman');hold on;
text(x(51-erty-4),b1(51,2)-0.04,num2str(x(51)),'FontSize',12,'FontName','Times New Roman');hold on;
text(x(60-erty),b1(60,2)+0.04,num2str(x(60)),'FontSize',12,'FontName','Times New Roman');hold on;
text(x(57-erty),b1(57,3)+0.04,num2str(x(57)),'FontSize',12,'FontName','Times New Roman');hold on;
text(x(73-erty),b1(73,3)+0.04,num2str(x(73)),'FontSize',12,'FontName','Times New Roman');hold on;
leg = legend('PC1:68.46%','PC2:28.94%','PC3:1.34%','PC4:0.45%','Location',[0.33 0.22 0.1 0.08],'FontSize',12,'Fontname', 'Times New Roman');
leg.ItemTokenSize = [15,18];
legend('boxoff');
text(600,0.52,str5,'FontSize',15)



subplot(3,2,6);
qwert=mean(C1(1:27,35:171));
plot(x,qwert,'k-','LineWidth',1);hold on;
pos=axis;
xlabel({'\fontname{Times New Roman}Wavelength (nm)'},'FontSize',12,'position',[pos(2)-600 pos(3)-0.1]);
ylabel({'\fontname{Times New Roman}Reflectance'},'FontSize',12);
% title({'\fontname{Times New Roman}Origin'},'FontSize',12);
xlim([500, 1010]); 
set(gca,'tickdir','out');
set(gca,'FontSize',12,'Fontname', 'Times New Roman');
set(gca, 'Box', 'off');
scatter(x(123),qwert(123),erty2,'k','filled');hold on;
scatter(x(16),qwert(16),erty2,'k','filled');hold on;
scatter(x(51),qwert(51),erty2,'k','filled');hold on;
scatter(x(60),qwert(60),erty2,'k','filled');hold on;
scatter(x(57),qwert(57),erty2,'k','filled');hold on;
scatter(x(73),qwert(73),erty2,'k','filled');hold on;
bar(x(123),qwert(123),qwert(123)*10,'FaceColor',[0 0.4470 0.7410]);hold on;
bar(x(16),qwert(16),qwert(16)*10,'FaceColor',[0.8500 0.3250 0.0980]);hold on;
bar(x(51),qwert(51),qwert(51)*10,'FaceColor',[0.8500 0.3250 0.0980]);hold on;
bar(x(60),qwert(60),qwert(60)*10,'FaceColor',[0.8500 0.3250 0.0980]);hold on;
bar(x(57),qwert(57),qwert(57)*10,'FaceColor',[0.9290 0.6940 0.1250]);hold on;
bar(x(73),qwert(73),qwert(73)*10,'FaceColor',[0.9290 0.6940 0.1250]);hold on;
text(600,0.55,str6,'FontSize',15)

for NN =1:6
    H(NN)=subplot(3,2,NN);%第NN张子图
    PPP=get(H(NN),'pos');%第NN张子图的当前位置
    PPP(1)=PPP(1)-0.01;%向右边延展0.04
    PPP(2)=PPP(2)-0.02;%向XIA方延展0.03
    PPP(3)=PPP(3)+0.01;%向右边延展0.04
    PPP(4)=PPP(4)+0.02;%向上方延展0.03
%     PPP(2)=PPP(2)-0.04;%向XIA方延展0.03
    set(H(NN),'pos',PPP);%根据新的边界设置。
    box off
    ax2 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k');
    set(ax2,'YTick', []);
    set(ax2,'XTick', []);
end

