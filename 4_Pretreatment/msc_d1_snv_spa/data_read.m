clear all;
clc
% 1、数据读取
% imgDataPath='F:\0626\else\';
% 
% imgDataDir  = dir([imgDataPath  '*']); 
% a=zeros(56,176);
% for op=3:length(imgDataDir)%遍历所有图片
%     a(op-2,:)=xlsread([imgDataDir(op).folder '\' imgDataDir(op).name '\Reflectance\Hyperspectral_data1.xlsx']);
% end

% 2、数据预览
xx = xlsread("D:\松针\songzhen\x.xls");

b = xlsread("D:\松针\songzhen\origin.xls");

ck_2 = b(1:8,:);
k1_2 = b(9:16,:);
k2_2 = b(17:25,:);
k3_2 = b(26:34,:);
n1_2 = b(35:41,:);
n2_2 = b(42:49,:);
n3_2 = b(50:54,:);
p1_2 = b(55:62,:);
p2_2 = b(63:72,:);
p3_2 = b(73:79,:);

ck_3 = b(80:89,:);
k1_3 = b(90:99,:);
k2_3 = b(100:107,:);
k3_3 = b(108:117,:);
n1_3 = b(118:126,:);
n2_3 = b(127:136,:);
n3_3 = b(137:142,:);
p1_3 = b(143:152,:);
p2_3 = b(153:162,:);
p3_3 = b(163:173,:);

ck_4 = b(174:182,:);
k1_4 = b(183:193,:);
k2_4 = b(194:203,:);
k3_4 = b(204:213,:);
n1_4 = b(214:220,:);
n2_4 = b(221:230,:);
n3_4 = b(231:239,:);
p1_4 = b(240:249,:);
p2_4 = b(250:259,:);
p3_4 = b(260:270,:);


b2=[ck_2;ck_3;ck_4;n1_2;n1_3;n1_4;n2_2;n2_3;n2_4;n3_2;n3_3;n3_4];

x = xx(1:176);x=x';
x2 = x(1:175);
X = b2(:,1:176);
Y = b2(:,177);
b1 = b2(:,1:176);
% 归一化
A1=mapminmax(b1,0,1);

% % 一阶导
A3=diff(b1');A3=A3';

% MSC
bb=mean(b1);
A6=msc(b1,bb);

str1 = {'\fontname{Arial}(a)'};
str2 = {'\fontname{Arial}(b)'};
str3 = {'\fontname{Arial}(c)'};
str4 = {'\fontname{Arial}(d)'};


h = figure;
set(h,'position',[100 100 400 300]);
% subplot(2,2,1)
plot(x(:),mean(b1(1:27,:)),'color',[0 0.4470 0.7410],'LineWidth',1.2);hold on;
plot(x(:),mean(b1(28:50,:)),'color',[0.8500 0.3250 0.0980],'LineWidth',1.2);hold on;
plot(x(:),mean(b1(51:78,:)),'color',[0.9290 0.6940 0.1250],'LineWidth',1.2);hold on;
plot(x(:),mean(b1(79:98,:)),'color',[0.4940 0.1840 0.5560],'LineWidth',1.2);hold on;
xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',12);
ylabel({'\fontname{Arial}Reflectance'},'FontSize',12);
set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1.2);
set(gca, 'Box', 'off');
axis([400 1000 0 0.6]);
box on
legend('\fontname{Arial}CK','\fontname{Arial}N1','\fontname{Arial}N2','\fontname{Arial}N3','Location','southeast','LineWidth',1.2);
% text(450,0.5,str1,'FontSize',12)
% 
% subplot(2,2,2)
% plot(x(37:171),mean(A1(1:27,37:171)),'color',[0 0.4470 0.7410],'LineWidth',1.2);hold on;
% plot(x(37:171),mean(A1(28:50,37:171)),'color',[0.8500 0.3250 0.0980],'LineWidth',1.2);hold on;
% plot(x(37:171),mean(A1(51:78,37:171)),'color',[0.9290 0.6940 0.1250],'LineWidth',1.2);hold on;
% plot(x(37:171),mean(A1(79:98,37:171)),'color',[0.4940 0.1840 0.5560],'LineWidth',1.2);hold on;
% xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',12);
% ylabel({'\fontname{Arial}Reflectance'},'FontSize',12);
% set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1.2);
% set(gca, 'Box', 'off');
% axis([500 1000 0 1]);
% box on
% legend('\fontname{Arial}CK','\fontname{Arial}N1','\fontname{Arial}N2','\fontname{Arial}N3','Location','southeast','LineWidth',1.2);
% text(550,0.85,str2,'FontSize',12)
% 
% subplot(2,2,3)
% plot(x(37:171),mean(A6(1:27,37:171)),'color',[0 0.4470 0.7410],'LineWidth',1.2);hold on;
% plot(x(37:171),mean(A6(28:50,37:171)),'color',[0.8500 0.3250 0.0980],'LineWidth',1.2);hold on;
% plot(x(37:171),mean(A6(51:78,37:171)),'color',[0.9290 0.6940 0.1250],'LineWidth',1.2);hold on;
% plot(x(37:171),mean(A6(79:98,37:171)),'color',[0.4940 0.1840 0.5560],'LineWidth',1.2);hold on;
% xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',12);
% ylabel({'\fontname{Arial}Reflectance'},'FontSize',12);
% set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1.2);
% set(gca, 'Box', 'off');
% axis([500 1000 0 0.6]);
% box on
% legend('\fontname{Arial}CK','\fontname{Arial}N1','\fontname{Arial}N2','\fontname{Arial}N3','Location','southeast','LineWidth',1.2);
% text(550,0.5,str3,'FontSize',12)
% 
% subplot(2,2,4)
% plot(x2(37:171),mean(A3(1:27,37:171)),'color',[0 0.4470 0.7410],'LineWidth',1.2);hold on;
% plot(x2(37:171),mean(A3(28:50,37:171)),'color',[0.8500 0.3250 0.0980],'LineWidth',1.2);hold on;
% plot(x2(37:171),mean(A3(51:78,37:171)),'color',[0.9290 0.6940 0.1250],'LineWidth',1.2);hold on;
% plot(x2(37:171),mean(A3(79:98,37:171)),'color',[0.4940 0.1840 0.5560],'LineWidth',1.2);hold on;
% xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',12);
% ylabel({'\fontname{Arial}Reflectance'},'FontSize',12);
% set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1.2);
% set(gca, 'Box', 'off');
% axis([500 1000 -0.02 0.06]);
% box on
% legend('\fontname{Arial}CK','\fontname{Arial}N1','\fontname{Arial}N2','\fontname{Arial}N3','Location','southeast','LineWidth',1.2);
% text(550,0.048,str4,'FontSize',12)

% for NN =1:4
%     H(NN)=subplot(2,2,NN);%第NN张子图
%     PPP=get(H(NN),'pos');%第NN张子图的当前位置
%     PPP(1)=PPP(1)-0.01;%向右边延展0.04
%     PPP(2)=PPP(2)+0.04;%向XIA方延展0.03
%     PPP(3)=PPP(3)+0.01;%向右边延展0.04
%     PPP(4)=PPP(4)+0.01;%向上方延展0.03
% %     PPP(2)=PPP(2)-0.04;%向XIA方延展0.03
%     set(H(NN),'pos',PPP);%根据新的边界设置。
%     box off
%     ax2 = axes('Position',get(gca,'Position'),...
%            'XAxisLocation','top',...
%            'YAxisLocation','right',...
%            'Color','none',...
%            'XColor','k','YColor','k');
%     set(ax2,'YTick', []);
%     set(ax2,'XTick', []);
% end

% for NN =1:2
%     H(NN)=subplot(2,2,NN);%第NN张子图
%     PPP=get(H(NN),'pos');%第NN张子图的当前位置
%     PPP(1)=PPP(1)-0.03;%向右边延展0.04
%     PPP(2)=PPP(2)-0.03;%向XIA方延展0.03
%     PPP(3)=PPP(3)-0.03;%向右边延展0.04
%     PPP(4)=PPP(4)+0.03;%向上方延展0.03
% %     PPP(2)=PPP(2)-0.04;%向XIA方延展0.03
%     set(H(NN),'pos',PPP);%根据新的边界设置。
% end