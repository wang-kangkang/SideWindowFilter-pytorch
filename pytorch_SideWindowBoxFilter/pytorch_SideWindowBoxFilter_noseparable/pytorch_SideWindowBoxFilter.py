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

        def get_kernel(a,b,r):
            a__b = (np.repeat(a, 2 * r + 1, axis = 1)) * (np.repeat(b.T ,2 * r + 1, axis=0))
            a__b[r,r] -= 1
            return a__b

        k_L__k_L = get_kernel(k_L,k_L,r)
        k_L__k_R = get_kernel(k_L,k_R,r)
        k_R__k_L = get_kernel(k_R,k_L,r)
        k_R__k_R = get_kernel(k_R,k_R,r)
        k_L__k = get_kernel(k_L,k,r)
        k_R__k = get_kernel(k_R,k,r)
        k__k_L = get_kernel(k,k_L,r)
        k__k_R = get_kernel(k,k_R,r)
        kernel = np.array([k_L__k_L, k_L__k_R, k_R__k_L, k_R__k_R, k_L__k, k_R__k, k__k_L, k__k_R]).astype(np.float32)
        kernel = kernel[:,np.newaxis,:,:]

        self.conv1 = nn.Conv2d(1,8,kernel_size=2*r+1,bias=False,padding=self.radius)
        self.conv1.weight = torch.nn.Parameter(torch.from_numpy(kernel))

    def forward(self, img):
        #prapare img
        img = np.float32(img)
        U = np.pad(img,((self.radius,self.radius),(self.radius,self.radius),(0,0)),'edge')
        for i in range(self.iteration):
             
            U_input = U
            U_input = U_input.transpose((2,0,1))
            U_input = U_input[:,np.newaxis,:,:]
            U_input = torch.from_numpy(U_input)
             
            #forward
            #d = self.conv1(U_input)
            d = self.conv1(U_input.cuda())
             
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
