import numpy as np
import torch
import torch.nn as nn
import cv2
import time

class SideWindowBoxFilter(nn.Module):
    def __init__(self, radius, iteration):
        super(SideWindowBoxFilter,self).__init__()

        self.radius = radius
        self.iteration = iteration
        r = radius
        k = np.ones((2 * r + 1, 1)) / (2 * r + 1) #separable kernel
        k_L = k.copy()
        k_L[r+1:]=0
        k_L = k_L/np.sum(k_L) #half kernel
        k_R = k_L[::-1,:]

        kernel1 = np.array([k_L,k_L,k_R,k_R,k_L,k_R,k,k]).astype(np.float32)
        kernel2 = np.array([k_L.T, k_R.T, k_L.T, k_R.T, k.T,k.T, k_L.T, k_R.T]).astype(np.float32)
        kernel1 = kernel1[:,np.newaxis,:,:]
        kernel2 = kernel2[:,np.newaxis,:,:]
        self.conv1 = nn.Conv2d(1,8,kernel_size=2*r+1,bias=False,padding=(self.radius,0))
        self.conv1.weight = torch.nn.Parameter(torch.from_numpy(kernel1))
        self.conv2 = nn.Conv2d(8,8,kernel_size=2*r+1,bias=False,padding=(0,self.radius),groups=8)
        self.conv2.weight = torch.nn.Parameter(torch.from_numpy(kernel2))

    def forward(self, img):
        #prapare img
        img = np.float32(img)
        U = np.pad(img,((self.radius,self.radius),(self.radius,self.radius),(0,0)),'edge')
        for i in range(self.iteration):
             
            U_input = U
            U_input = U_input.transpose((2,0,1))
            U_input = U_input[:,np.newaxis,:,:]
            U_input = torch.from_numpy(U_input)
            U_input = U_input.cuda()
             
            #forward
            output1 = self.conv1(U_input)
            output2 = self.conv2(output1)
            d = output2 - U_input
             
            #abs and index
            d_abs = torch.abs(d)
            dm = torch.min(d_abs, dim = 1, keepdim = True)
            dm_index = dm[1]
            dm = torch.gather(d,1,dm_index)

            #get and return
            dm = dm.cpu().detach().numpy()
            dm = dm[:,0,:,:]
            dm = dm.transpose((1,2,0))
            U = U + dm
             
        U = U[self.radius:-self.radius,self.radius:-self.radius,:]
        return U
        
if __name__=='__main__':
    model = SideWindowBoxFilter(6, 30)
    model = model.cuda()
    model.eval()
    img = cv2.imread('./lena.bmp').astype(np.float32)
    img = img/256
    time_start = time.time()
    out = model(img)
    time_end = time.time()
    print("time:%f2"%(time_end-time_start))
    cv2.imwrite('py_out.bmp',out*256)
    for i in range(3):
        np.savetxt('aa'+str(i)+'.txt',out[:,:,i])
    print('end')
