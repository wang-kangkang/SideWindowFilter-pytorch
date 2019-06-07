img = imread('./lena.bmp');
img = double(img)/256;
tic
result=SideWindowBoxFilter(img, 6, 30);
toc
%result=SideWindowBoxFilter_origin(img, 5, 3);

pytorch_result = zeros(size(result));
for i=1:3
    pytorch_result(:,:,i) = importdata(['aa',num2str(3-i),'.txt']);
    figure;
    mesh(pytorch_result(:,:,i)-result(:,:,i));
    title('diff')
end
