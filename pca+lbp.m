%LBP+PCA test: 以LBP描述子（直方图）作为输入特征，降维，最近邻分类器识别系统实验

close all; clear all; clc;

tic;

 

%============================①读取训练集和测试集============================%

 

training_set = [];            %创建一个空白数组，用于存放所有训练集图像

for i = 1:40

    for j = 1:5               %只读取每个人的前五张图片作为训练图像

       a = imread(strcat('s',num2str(i),'\',num2str(j),'.bmp'));

       b = a(1:112*92);      %将以上数据转换为1×N的行向量存入b，N=112×92=10304

                             %提取顺序是从上到下，从左到右

       b = double(b);        %强制数据类型转换，便于后面的运算操作

       training_set = [training_set;b];

       %每循环一次，就在原来的traing_set后面添加一行

       %traing_set是一个M×N矩阵，每一行代表一张图片。M=40×5=200, N=10304

    end

end

 

testing_set = [];             %创建一个空白数组，用于存放所有测试集图像

for i = 1:40

    for j = 6:10              %读取每个人的后五张图片作为测试图像

       a = imread(strcat('s',num2str(i),'\',num2str(j),'.bmp'));

       b = a(1:112*92);      %将以上数据转换为1×N的行向量存入b，N=112×92=10304

                              %提取顺序是从上到下，从左到右

       b = double(b);        %强制数据类型转换，便于后面的运算操作

       testing_set = [testing_set;b];

       %每循环一次，就在原来的testing_set后面添加一行

       %testing_set是一个M×N矩阵，每一行代表一张图片。M=40×5=200, N=10304

    end

end

 

%===================②转化为LBP图像，分块，提取直方图，连接===================%

 

%-------------------------------------------

%准备存放训练集输入特征

training_f = [];

 

for I = 1:200

    X =reshape(training_set(I,:),112,92);   %将训练集每一行数据先恢复成矩阵形式

   

    [m,n]= size(X);                         %读取图片尺寸

    X =double(X);                           %转换成double类型，后面才能参与运算

    extend= zeros(m+2,n+2);                 %扩展一圈的全零矩阵

    extend(2:m+1,2:n+1)= X;                 %把X放到extend中间

 

    extend(1,2:n+1)= X(1,1:n);              %向上扩展

    extend(m+2,2:n+1)= X(m,1:n);            %向下扩展

    extend(2:m+1,1)= X(1:m,1);              %向左扩展

    extend(2:m+1,n+2)= X(1:m,n);            %向右扩展

 

    B2D= [1 2 4;128 0 8;64 32 16];          %二进制→十进制转换模板

 

    Result= zeros(m,n);

    for i = 1:m

       for j = 1:n

           temp =  X(i,j);                  %取某点的像素值，即中心阈值a

           A = ones(3)*temp;                %3*3矩阵，元素全为a

           B = extend(i:(i+2),j:(j+2));     %从扩展图像中取3*3邻域矩阵

           C = B - A;                       %相减

           C(find(C>=0)) = 1;              %判断C中非负数，置1

           C(find(C<0)) = 0;               %判断C中负数，置0

           Result(i,j) = sum(sum(C.*B2D));  %点乘求和，返回Result作为LBP值

       end

    end

    %这里得到的Result就是112*92的LBP图像

   

    H =[]; %预先准备一个空白矩阵，用于存放直方图数据

    for p = 1:7

       for q = 1:7

           Block = zeros(16,13);

           Block = Result(((p-1)*16+1):((p-1)*16+16),((q-1)*13+1):((q-1)*13+13));

           %读取第p行第q列的分块

           Block = reshape(Block,1,208);    %转换成行向量才能求直方图

           h = hist(Block,255);             %直方图存入h（1*255行向量）

           H = [H;h];                       %每循环一次，就在原来的H后面添加一行

                                             %H是49*255矩阵

       end

    end

   

    training_f= [training_f; H(1:49*255)];

    %每循环一次(开头的大I=1:200的循环)，就在training_f后面添加一行

end

 

 

%-------------------------------------------

%准备存放测试集输入特征

testing_f = []; 

 

for I = 1:200

    X =reshape(testing_set(I,:),112,92);   %将测试集每一行数据先恢复成矩阵形式

   

    [m,n]= size(X);                         %读取图片尺寸

    X =double(X);                           %转换成double类型，后面才能参与运算

    extend= zeros(m+2,n+2);                 %扩展一圈的全零矩阵

    extend(2:m+1,2:n+1)= X;                 %把X放到extend中间

 

    extend(1,2:n+1)= X(1,1:n);              %向上扩展

    extend(m+2,2:n+1)= X(m,1:n);            %向下扩展

    extend(2:m+1,1)= X(1:m,1);              %向左扩展

    extend(2:m+1,n+2)= X(1:m,n);            %向右扩展

 

    B2D= [1 2 4;128 0 8;64 32 16];          %二进制→十进制转换模板

 

    Result= zeros(m,n);

    for i = 1:m

       for j = 1:n

           temp =  X(i,j);                  %取某点的像素值，即中心阈值a

           A = ones(3)*temp;                %3*3矩阵，元素全为a

           B = extend(i:(i+2),j:(j+2));     %从扩展图像中取3*3邻域矩阵

           C = B - A;                       %相减

           C(find(C>=0)) = 1;              %判断C中非负数，置1

            C(find(C<0)) = 0;                %判断C中负数，置0

           Result(i,j) = sum(sum(C.*B2D));  %点乘求和，返回Result作为LBP值

       end

    end

    %这里得到的Result就是112*92的LBP图像

   

    H =[]; %预先准备一个空白矩阵，用于存放直方图数据

    for p = 1:7

       for q = 1:7

           Block = zeros(16,13);

           Block = Result(((p-1)*16+1):((p-1)*16+16),((q-1)*13+1):((q-1)*13+13));

           %读取第p行第q列的分块

           Block = reshape(Block,1,208);    %转换成行向量才能求直方图

           h = hist(Block,255);             %直方图存入h（1*255行向量）

            H = [H;h];                       %每循环一次，就在原来的H后面添加一行

                                             %H是49*255矩阵

       end

    end

   

    testing_f= [testing_f; H(1:49*255)];

    %每循环一次(开头的大I=1:200的循环)，就在training_f后面添加一行

end

 

 

 

%======================③利用PCA对上面得到的特征进行降维======================%

 

%注释直接沿用了原来的代码中的注释，不完全准确

 

Mean = mean(training_f);   

%计算平均值，得到一个1×N的行向量

X = [];   %把之前用过的X清空，重新利用。（否则会只有0.025）

 

for i=1:200

    X(i,:)=training_f(i,:)-Mean;

    %每个图片的数据都减去均值，实现训练集数据的均值归零（中心化）

    %减号两边的数据类型必须相同，原始数据的像素灰度值为整数，而平均值是浮点数，

    ...所以前面要转换。

    %X是M×N矩阵，每一行保存的数据是：每个原始图片 - 平均图片

end

 

k=15;                       %暂时把维数选为15，即采用影响最大的15个特征

Sigma=X*X';                 %计算“协方差矩阵”，M×M矩阵

[V,D]=eigs(Sigma,k);        %计算“协方差矩阵”的前k个特征值和特征向量

E=X'*V;                    %计算特征脸矩阵E，每一列是一张特征脸，共k列

 

Train_E=training_f*E;    

%训练集数据降维，得到M×k矩阵（M=200,k=15），每行代表一张图片

%原来一张图片需要用N=112*92个数字表示，现在用k个数字来代替

 

Test_E=testing_f*E;

%测试集数据降维，得到M×k矩阵（M=200,k=15），每行代表一张图片

 

 

 

 

 

%=============== ④比较降维后的训练集和样本集数据，得分类结果 ================%

 

correct=0;                  %初始化判断正确的人脸数

 

for i=1:200

    for j=1:200

       error(j)=norm(Test_E(i,:)-Train_E(j,:));

       %将Test_E的第i行与Train_E的每一列求距离（即：相减，再取2-范数）

    end

    [errormin,I]=min(error);

    %算得距离最小的是第I行

   

    true_class=ceil(i/5);   %实际上我知道当前这一行的数据是代表第几个人

    recog_class=ceil(I/5);  %而我让系统算出来是第几个人

   

    if true_class==recog_class

       correct=correct+1;  %如果系统判断正确，那么得一分~

    end

   

end

 

accuracy=correct/200        %输出正确率

 

toc;