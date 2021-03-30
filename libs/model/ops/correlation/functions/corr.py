import torch
from torch.autograd import Function
import correlation_cuda
import correlation_proj


class correlation_op(Function):
    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        b, c, h, w = input1.size()
        ctx.save_for_backward(input1, input2)
        h_pad = 2*pad_size+h
        w_pad = 2*pad_size+w

        width = 2*max_displacement//stride2+1
        rbot1 = torch.zeros(b, c, h_pad, w_pad, device=input1.device)
        rbot2 = torch.zeros(b, c, h_pad, w_pad, device=input1.device)
        output = torch.zeros(b, width*width, h, w, device=input1.device)

        correlation_cuda.corr_cuda_forward(input1, input2,
                                           rbot1, rbot2,
                                           output,
                                           pad_size,
                                           kernel_size,
                                           max_displacement,
                                           stride1,
                                           stride2,
                                           corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input1, input2 = ctx.saved_tensors
        b, c, h, w = input1.size()
        h_pad = 2*ctx.pad_size+h
        w_pad = 2*ctx.pad_size+w
        rbot1 = torch.zeros(b, c, h_pad, w_pad, device=input1.device)
        rbot2 = torch.zeros(b, c, h_pad, w_pad, device=input1.device)

        grad_input1 = torch.zeros(input1.size(), device=input1.device)
        grad_input2 = torch.zeros(input2.size(), device=input1.device)


        correlation_cuda.corr_cuda_backward(input1, input2,
                                            rbot1, rbot2,
                                            grad_output,
                                            grad_input1,
                                            grad_input2,
                                            ctx.pad_size,
                                            ctx.kernel_size,
                                            ctx.max_displacement,
                                            ctx.stride1,
                                            ctx.stride2,
                                            ctx.corr_multiply)
        return grad_input1, grad_input2, None, None, None, None, None, None


class CorrProj(Function):
    @staticmethod
    def forward(ctx, input1, input2, query_coords, scene_coords, scene_P, max_displacement, stride):
        ctx.max_displacement = max_displacement
        ctx.stride = stride

        
        x = correlation_proj.corr_proj_forward(input1, input2,
                                               query_coords,
                                               scene_coords,
                                               scene_P,
                                               max_displacement,
                                               stride)
        ctx.save_for_backward(input1, input2, query_coords, scene_P)
        return x[0], x[1], x[2]

    @staticmethod
    def backward(ctx, grad_output1, grad_output2, grad_output3):
        input1, input2, query_coords, scene_P= ctx.saved_tensors
        max_displacement = ctx.max_displacement
        stride = ctx.stride
        
        grad_output1=grad_output1.contiguous()
        x = correlation_proj.corr_proj_backward(grad_output1,
                                               input1,
                                               input2,
                                               query_coords, scene_P, max_displacement, stride
                                               )
        return x[0],x[1], None, None, None, None, None


correlation_op = correlation_op.apply
correlation_proj_op = CorrProj.apply
