"""
reference:
    positional encoding:
        https://github.com/tatp22/multidim-positional-encoding
    concatenate torch:
        https://pytorch.org/docs/stable/generated/torch.cat.html
    nested for loop in one line:
        https://easytoread.tistory.com/entry/Python-%EC%9D%B4%EC%A4%91-for%EB%AC%B8-%EB%A6%AC%EC%8A%A4%ED%8A%B8-%EC%BB%B4%ED%94%84%EB%A6%AC%ED%95%B8%EC%85%98
    
"""

import torch
import torch.nn as nn
import numpy as np

from PIL import Image


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size(), penc.size()
        )
        return tensor + penc


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        # print("shape of input tensor:\n\t",tensor.shape)
        tensor = tensor.permute(0, 2, 3, 1)
        # print("shape of tensor after permutation:\n\t",tensor.shape)
        enc = self.penc(tensor)
        # print("shape of tensor after positional encoding:\n\t",tensor.shape)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels
    

def positional_encoding(args, data):
    # print("shape of input data:\n\t",data.shape)
    
    ## data [batch_size, 1, 512, 512]
    data_org = data[:]

    ## positional encoding [batch_size, 10, 512, 512]
    data_pe = data[:]
    for i in range(9):
        data_pe = torch.cat((data_pe,data_org),1)
    pe2dSum = Summer(PositionalEncodingPermute2D(10))
    data_pe = pe2dSum(data_pe)
    # print("shape of data_pe:\n\t",data_pe.shape)

    ## (x,y) [batch_size, 2, 512, 512]
    x = torch.Tensor([[i]*args.image_resize for i in range(args.image_resize)])
    y = torch.Tensor([[j for j in range(args.image_resize)] for i in range(args.image_resize)])
    x, y = x.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0)
    x_tmp, y_tmp = x[:], y[:]
    for i in range(data.size()[0]-1):
        x = torch.cat((x,x_tmp),0)
        y = torch.cat((y,y_tmp),0)
    # print("shape of x:\n\t",x.shape)
    # print("shape of y:\n\t",y.shape)

    regenerated_data = torch.cat((data_org,data_pe),1)
    # print("shape of regenerated_data1 :\n\t",regenerated_data.shape)
    regenerated_data = torch.cat((regenerated_data,x),1)
    # print("shape of regenerated_data2 :\n\t",regenerated_data.shape)
    regenerated_data = torch.cat((regenerated_data,y),1)
    # print("shape of regenerated_data3 :\n\t",regenerated_data.shape)

    return regenerated_data


# pe2d = PositionalEncodingPermute2D(10)
# tmp = torch.zeros(12,512,512)
# pe2d(tmp)
# print(pe2d(tmp).shape)
# print(tmp.shape)
# tmp = tmp.unsqueeze(1)
# print(tmp.shape)

# pe2dSum = Summer(PositionalEncodingPermute2D(10))
# pe2dSum(tmp)
# print(pe2dSum(tmp).shape)

# a = torch.zeros(12,1,512,512)
# b = a[:]

# print(torch.cat((a,b),1).shape)

# for i in range(9):
#     a = torch.cat((a,b),1)
#     print(a.shape)

# # a = pe2dSum(a)
# # print(a.shape)

# penc_no_sum = pe2d(a) # penc_no_sum.shape == (1, 6, 10)
# penc_sum = pe2dSum(a)
# # print(penc_no_sum + a == penc_sum) # True



# image_path = "/home/yehyun/2022-Fall-Research/data/padded_image/0_pad.png"
# image = np.array(Image.open(image_path).convert("L").resize((512,512)))
# print("original image shape: \n\t",image.shape)

# # image = np.transpose(image, (2,0,1))
# image = torch.Tensor(image).unsqueeze(0)
# image = torch.Tensor(image).unsqueeze(0)
# print("after unsqueeze:\n\t",image.shape)

# p_enc_2d = PositionalEncodingPermute2D(10)
# print("positional encoding:\n\t",p_enc_2d(image).shape)

# penc_no_sum = p_enc_2d(image)
# p_enc_2d_model_sum = Summer(PositionalEncodingPermute2D(10))
# penc_sum = p_enc_2d_model_sum(image)

# print(image.shape, penc_no_sum.shape, penc_sum.shape)

# label = [19,246,303,206,301,240,331,189,329,239,500,243]
# # print(model(torch.Tensor(image)))
# # print(model(image))

# x = torch.Tensor([[i]*512 for i in range(512)])
# print(x.shape)
# x = x.unsqueeze(0).unsqueeze(0)
# x_tmp = x[:]
# for i in range(11):
#     x = torch.cat((x,x_tmp),0)
# print(x.shape)

# y = torch.Tensor([[j for j in range(512)] for i in range(512)])
# print(y.shape)
# print(y)
