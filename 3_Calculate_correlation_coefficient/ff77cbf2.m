clear all;
clc
SamplePath =  'H:\肇庆松针N素\20210507台山实验_5高光谱相机\3\';  %存储图像的路径
x = xlsread('D:\松针\x.xlsx');

imgDataDir  = dir([SamplePath '*']); 

for tt=3:272
imagefile=[imgDataDir(tt).folder '\' imgDataDir(tt).name  '\' ];

fileExt = '*.bmp';  %待读取图像的后缀名
%获取所有路径
files = dir(fullfile(imagefile,fileExt)); 
len1 = size(files,1);
%遍历路径下每一幅图像
files1 = files(3:176,:);
files2 = files(1:2,:);
files3 = [files1; files2]; 

for i=1:len1-1
    fileName = strcat(imagefile,'\',files3(i).name); 
	image = imread(fileName);
    

    fileName2 = strcat(imagefile,'\',files3(i+1).name); 
    image2 = imread(fileName2);    

    %读入两幅图像
    imgA=image;
    imgB=image2;

    %精度转换
    imgA=double(imgA);
    imgB=double(imgB);

    %健壮性判断
    [imgArow,imgAcol]=size(imgA);
    [imgBrow,imgBcol]=size(imgB);
    if imgArow<1||imgAcol<1||imgBrow<1||imgBcol<1
        error('您的输入有误！维数不能小于1');
    elseif imgArow~=imgBrow||imgAcol~=imgBcol
        error('您输入的矩阵维数不相等！');
    end
    %求两幅图像的均值差
    imgA=imgA-mean2(imgA);
    imgB=imgB-mean2(imgB);

    %求求两幅图像的相关系数
    CC(i,tt-2)=sum(sum(imgA.*imgB))./(sqrt(sum(sum(imgA.^2))).*sqrt(sum(sum(imgB.^2))));

end

end
% 
