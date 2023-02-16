 %function read_image_raw(hdrfile,imagefile,savefold)
clear all;
clc
hdrfile='D:\松针\songzhen\related_coefficient\header.hdr';
imgDataPath='H:\肇庆松针N素\20210507台山实验_5高光谱相机\2、反射率数据\';
imgDataPath2='H:\广藿香\3\';

imgDataDir  = dir([imgDataPath '*.raw']); 



for tt=1:length(imgDataDir)
mkdir([imgDataPath2,imgDataDir(tt).name]);

imagefile=imgDataDir(tt).name;
savefold=[imgDataPath2 imgDataDir(tt).name '\'];

hdrfilename=hdrfile;
imagefilename=[imgDataPath imagefile];
savefolder=savefold;
fid = fopen(hdrfilename,'r');
info = fread(fid,'char=>char');
info=info';%默认读入列向量，须要转置为行向量才适于显示
fclose(fid);
%查找列数
a=strfind(info,'samples = ');
b=length('samples = ');
c=strfind(info,'lines');
samples=[];
for i=a+b:c-1
samples=[samples,info(i)];
end
samples=str2num(samples);
%查找行数
a=strfind(info,'lines = ');
b=length('lines = ');
c=strfind(info,'bands');
lines=[];
for i=a+b:c-1
lines=[lines,info(i)];
end
lines=str2num(lines);
%查找波段数
a=strfind(info,'bands = ');
b=length('bands = ');
c=strfind(info,'header offset');
bands=[];
for i=a+b:c-1
bands=[bands,info(i)];
end
bands=str2num(bands);
%查找波长数目
a=strfind(info,'wavelength = {');
b=length('wavelength = {');
% c=strfind(info,' }');
wavelength=[];
for i=a+b:length(info)
    wavelength=[wavelength,info(i)];
end
wave=sscanf(wavelength,'%f,');
%查找数据类型
a=strfind(info,'data type = ');
b=length('data type = ');
c=strfind(info,'interleave');
datatype=[];
for i=a+b:c-1
datatype=[datatype,info(i)];
end
datatype=str2num(datatype);
precision=[];
switch datatype
case 1
precision='uint8=>uint8';%头文件中datatype=1对应ENVI中数据类型为Byte，对应MATLAB中数据类型为uint8
case 2
precision='int16=>int16';%头文件中datatype=2对应ENVI中数据类型为Integer，对应MATLAB中数据类型为int16
case 12
precision='uint16=>uint16';%头文件中datatype=12对应ENVI中数据类型为Unsighed Int，对应MATLAB中数据类型为uint16
case 3
precision='int32=>int32';%头文件中datatype=3对应ENVI中数据类型为Long Integer，对应MATLAB中数据类型为int32
case 13
precision='uint32=>uint32';%头文件中datatype=13对应ENVI中数据类型为Unsighed Long，对应MATLAB中数据类型为uint32
case 4
precision='float32=>float32';%头文件中datatype=4对应ENVI中数据类型为Floating Point，对应MATLAB中数据类型为float32
case 5
precision='double=>double';%头文件中datatype=5对应ENVI中数据类型为Double Precision，对应MATLAB中数据类型为double
otherwise
error('invalid datatype');%除以上几种常见数据类型之外的数据类型视为无效的数据类型
end
%查找数据格式
a=strfind(info,'interleave = ');
b=length('interleave = ');
c=strfind(info,'sensor type');
interleave=[];
for i=a+b:c-1
interleave=[interleave,info(i)];
end
interleave=strtrim(interleave);%删除字符串中的空格
%读取图像文件
fid = fopen(hdrfilename, 'r');
data = multibandread(imagefilename,[lines samples bands],precision,0,interleave,'ieee-le');
data= double(data);
for i=1:bands
    save_path=strcat(savefolder,num2str(wave(i)),'.bmp');
    imwrite(data(:,:,i),save_path);
end
% ppt = data(:,:,56);
% 保存图像
end