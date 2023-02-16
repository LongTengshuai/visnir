wer = xlsread("C:\Users\Administrator\Desktop\20220929.xlsx");
x=400:3.43:1000.25;
% 归一化
A1=mapminmax(wer,0,1);

A2=diff(wer');A2=A2';

% MSC
bb=mean(wer);
A3=msc(wer,bb);

figure
plot(x,wer(1:20,:),'r');hold on;
plot(x,wer(21:40,:),'g');hold on;
plot(x,wer(41:60,:),'b');hold on;

xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',12);
ylabel({'\fontname{Arial}Reflectance'},'FontSize',12);
set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1.2);
set(gca, 'Box', 'off');
axis([400 1000 0 0.6]);
box on

figure
plot(x(1:175),A2(1:20,:),'r');hold on;
plot(x(1:175),A2(21:40,:),'g');hold on;
plot(x(1:175),A2(41:60,:),'b');hold on;

xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',12);
ylabel({'\fontname{Arial}Reflectance'},'FontSize',12);
set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1.2);
set(gca, 'Box', 'off');
axis([400 1000 -0.05 0.05]);
box on

figure
plot(x,A3(1:20,:),'r');hold on;
plot(x,A3(21:40,:),'g');hold on;
plot(x,A3(41:60,:),'b');hold on;

xlabel({'\fontname{Arial}Wavelength (nm)'},'FontSize',12);
ylabel({'\fontname{Arial}Reflectance'},'FontSize',12);
set(gca,'FontSize',12,'Fontname', 'Arial','LineWidth',1.2);
set(gca, 'Box', 'off');
axis([400 1000 0 0.6]);
box on