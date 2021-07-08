import torch
from net import Net
from utils import get_argument_parser, set_seed, create_folder, check_memory
from plot import plot_coefficient_maps, plot_pred_map
from preprocess import data_preprocessing
import numpy as np
import sys, os

def load_net_and_test(args, device):
    _, _, _, X_mesh, n_in, u, s, \
    v, nS, time_mean, n_eofs  =  data_preprocessing(args)
    net = Net(n_in, n_eofs, v, s, nS, time_mean).to(device)
    ## Load Model / key problem
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.net).items()})
    ### Output Test
    with torch.no_grad():
        si = 0
        for i, ei in enumerate(range(0,X_mesh.shape[0]+1,50000)):
            #print("index :", si, ei)
            y_hat_mesh, y_pred_mesh = net(torch.from_numpy(X_mesh[si:ei]).type(torch.FloatTensor).to(device))
            if i == 0:
                y_hat_mesh_ = y_hat_mesh.clone().detach()
                y_pred_mesh_ = y_pred_mesh.clone().detach()
            else:
                y_hat_mesh_ = torch.cat((y_hat_mesh_,y_hat_mesh))
                y_pred_mesh_ = torch.cat((y_pred_mesh_, y_pred_mesh))
            si = ei
        ### last index
        ei = X_mesh.shape[0]+1
        y_hat_mesh, y_pred_mesh = net(torch.from_numpy(X_mesh[si:ei]).type(torch.FloatTensor).to(device))
        y_hat_mesh_ = torch.cat((y_hat_mesh_,y_hat_mesh))
        y_pred_mesh_ = torch.cat((y_pred_mesh_, y_pred_mesh))
    ### array
    if device == "cuda":
        y_hat_mesh_ = y_hat_mesh_.cpu()
        y_pred_mesh_= y_pred_mesh_.cpu()
    y_hat_mesh = y_hat_mesh_.clone().detach().numpy()
    y_pred_mesh = y_pred_mesh_.clone().detach().numpy()
    return y_hat_mesh, y_pred_mesh

def minmaxNorm(dt, minval, maxval):
  normed_dt = dt * (maxval - minval) + minval
  return normed_dt


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    create_folder(args.outf)
    #sys.stdout = open("%s/report_test.txt" %(args.outf),'w')
    check_memory("Start")

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    set_seed(args.manualSeed)

    y_hat_mesh, y_pred_mesh = load_net_and_test(args, device)

    if args.var == 'REH':
        y_hat_mesh[np.where(y_hat_mesh<0)]=0
        y_hat_mesh[np.where(y_hat_mesh>100)]=100
    else:
        pass
    #y_hat_mesh = minmaxNorm(y_hat_mesh, -50, 50)
    np.save('%s/y_hat_mesh.npy'%(args.outf), y_hat_mesh)
    np.save('%s/y_pred_mesh.npy'%(args.outf), y_pred_mesh)

    #check_memory("Test")
    #plot_coefficient_maps(args, y_pred_mesh)
    #plot_pred_map(args, y_hat_mesh)
    #check_memory("Finish")
    sys.stdout.close()

if __name__ == "__main__":
    main()
