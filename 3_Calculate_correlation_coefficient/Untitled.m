sd = xlsread('D:\松针\songzhen\related_coefficient\新建 Microsoft Excel 工作表.xlsx');

x = xlsread('C:\Users\Administrator\Desktop\spad\x.xls');
xx=ones(176,1);
str1 = {'\fontname{Arial}505.0nm'};
str2 = {'\fontname{Arial}986.8nm'};

h = figure;
set(h,'position',[100 100 600 300]);
subplot(1,1,1);
plot(x,x,'k','LineWidth',1.2);hold on;
scatter(x(33),0.71,'r','filled');hold on;
plot(x,0.71*xx,'k--','LineWidth',1.2);hold on;
plot(x(2:176),sd,'LineWidth',1.2);hold on;
scatter(x(33),0.71,'r','filled');hold on;

scatter(x(171),0.71,'r','filled');hold on;
xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',12);
ylabel({'\fontname{Arial}Correlation Coefficient'},'FontSize',12);
set(gca,'FontName','Arial','FontSize',12,'LineWidth',1.2);
set(gca, 'Box', 'off');
text(520,0.61,str1,'FontSize',12);
text(880,0.61,str2,'FontSize',12)
axis([400 1010 0 1]);
legend('\fontname{Arial}The correlation coefficient = 0.7','\fontname{Arial}Correlation coefficient between hyperspectral images');
box on
% ax2 = axes('Position',get(gca,'Position'),...
%     'Color','none',...
%     'XAxisLocation','top',...
%     'YAxisLocation','right',...
%     'XColor','k','YColor','k');
% set(ax2,'YTick', []);
% set(ax2,'XTick', []);

% for NN =1:1
%     H(NN)=subplot(1,1,NN);%第NN张子图
%     PPP=get(H(NN),'pos');%第NN张子图的当前位置
%     PPP(1)=PPP(1);%向右边延展0.04
%     PPP(2)=PPP(2)+0.08;%向XIA方延展0.03
%     PPP(3)=PPP(3)+0.03;%向右边延展0.04
%     PPP(4)=PPP(4)-0.08;%向上方延展0.03
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


