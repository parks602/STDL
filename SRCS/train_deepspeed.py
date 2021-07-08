# Loading all the required libraries.
import numpy as np
import pandas as pd
import os, sys
import deepspeed
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn    
from tqdm import trange
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
from utils import get_argument_parser, set_seed, create_folder
from preprocess import data_preprocessing, read_data, decomposition
from cmaps import COLORBARS
from net import Net
from plot import plot_learningCurve, plot_coefficient_maps, plot_compare_true_pred, plot_pred_map

def train_valid(args, train_loader, valid_loader, n_in, n_eofs, v, s, nS, time_mean, device):
    writer = SummaryWriter(log_dir=args.outf)

    v = v.to(device)
    s = s.to(device)
    nS = nS.to(device)
    time_mean = time_mean.to(device)

    net = Net(n_in, n_eofs, v, s, nS, time_mean).to(device)  ## he input are the geographical variables X, Y, [Z]
    summary(net, (n_in,))
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr, betas=(args.beta1, 0.999))
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader)-1, epochs=args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader)-1, epochs=args.epochs, anneal_strategy='cos')
    model, optimizer, _, lr_scheduler = deepspeed.initialize(args=args, model=net, model_parameters=net.parameters(), optimizer=optimizer, lr_scheduler=scheduler)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.outf)
    # epoch당 average training loss를 track
    avg_train_losses = []
    # epoch당 average validation loss를 track
    avg_valid_losses = []
    
    with trange(1, args.epochs + 1, desc='epoch') as epochs:
        for epoch in epochs:
            # 모델이 학습되는 동안 trainning loss를 track
            train_losses = []
            # 모델이 학습되는 동안 validation loss를 track
            valid_losses = []
            ### Train
            model.train()
            for batch, (x_data, y_data, u_data) in enumerate(train_loader,0): ## batch start 0
                # make grad zero before backward
                optimizer.zero_grad()
                # calculate loss
                x_data, y_data, u_data = x_data.to(device), y_data.to(device), u_data.to(device)
                pred_y, aux_y = net(x_data)
                loss_y = criterion(y_data, pred_y)
                loss_u = criterion(u_data, aux_y)
                loss = loss_y * 1.0 + loss_u * 0.0
                # backward
                model.backward(loss)
                # update
                model.step()
                train_losses.append(loss.item())
                writer.add_scalar("Train_Loss", loss.item(), epoch*len(train_loader)+batch)
            ### Valid
            with torch.no_grad():
                model.eval()
                for batch, (x_val, y_val, u_val) in enumerate(valid_loader,0):
                    x_val, y_val, u_val = x_val.to(device), y_val.to(device), u_val.to(device)
                    pred_y, aux_y = net(x_val)
                    print(pred_y, y_val, u_val, aux_y)
                    val_loss = criterion(y_val, pred_y)
                    valid_losses.append(val_loss.item())
                    writer.add_scalar("Valid_Loss", val_loss.item(), epoch*len(valid_loader)+batch)
            # epoch당 평균 loss 계산
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            # tqdm
            epochs.set_postfix(loss=train_loss, val_loss=valid_loss)
            # update scheduler
            lr_scheduler.step()
            ### Ealry Stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    model.load_state_dict(torch.load("%s/checkpoint.pt" %(args.outf)))
    # Plot LearningCurve
    plot_learningCurve(args, avg_train_losses, avg_valid_losses)
    return net, avg_train_losses, avg_valid_losses

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9988' # modify if RuntimeError: Address already in use
    os.environ['RANK'] = "0"
    os.environ['LOCAL_RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    print('strat')
    parser = get_argument_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    create_folder(args.outf)
    #sys.stdout = open("%s/report.txt" %(args.outf),'w')
    print(args)
    cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    set_seed(args.manualSeed)
    ### preprocess
    train_loader, valid_loader, X_test, X_mesh, n_in,\
    u, s, v, nS, time_mean, n_eofs =  data_preprocessing(args)
    #sys.exit()
    net, avg_train_losses, avg_valid_losses = train_valid(args, train_loader, valid_loader, n_in, n_eofs, v, s, nS, time_mean, device)
    y_hat_test, y_pred_test = net(torch.from_numpy(X_test).type(torch.FloatTensor).to(device))
    y_hat_test = y_hat_test.cpu().detach().numpy()
    '''
    ### Output
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
    y_hat_mesh_ = y_hat_mesh_.cpu()
    y_pred_mesh_= y_pred_mesh_.cpu()
    y_hat_mesh = y_hat_mesh_.clone().detach().numpy()
    y_pred_mesh = y_pred_mesh_.clone().detach().numpy()
    ### Plot
    plot_compare_true_pred(args, y_hat_mesh, y_hat_test)
    plot_coefficient_maps(args, y_pred_mesh)
    plot_pred_map(args, y_hat_mesh)
    sys.stdout.close()
    '''

if __name__ == "__main__":
    main()
