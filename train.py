import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import FSIterator

import numpy as np
import pandas as pd


def train_main(args, model, train_path, criterion, optimizer):
    iloop=0
    current_loss =0
    all_losses = [] 
    batch_size = args.batch_size
    train_iter = FSIterator(train_path, batch_size)
 
    for input,target, mask in train_iter: #TODO for debugging
        loss = train(args, model, input, mask, target, optimizer, criterion)
        current_loss += loss
        
        if (iloop+1) % args.logInterval == 0:
            print("%d %.4f" % (iloop+1, current_loss/args.logInterval))
            all_losses.append(current_loss /args.logInterval)
            current_loss=0
            break

        iloop+=1


def train(args, model, input, mask, target, optimizer, criterion):
    model = model.train()
    loss_matrix = []
    optimizer.zero_grad()

    output = model(input)
    
    for t in range(input.size(0)-1):
        loss = criterion(output[t].view(args.batch_size,-1), input[t+1][:,0].view(args.batch_size,-1))
        loss_matrix.append(loss.view(1,-1))

    loss_matrix = torch.cat(loss_matrix, dim=0)

    masked = loss_matrix * mask[1:]

    loss = torch.sum(masked) / torch.sum(mask[1:])

    loss.backward()

    optimizer.step()

    return loss.item()



def evaluate(args, model, input, mask, target, criterion):
    loss_matrix = []
    daylen = args.daytolook
    output = model(input)


    '''Part of loss'''
    for t in range(input.size(0)-1):
        loss = criterion(output[t].view(args.batch_size,-1), input[t+1][:,0].view(args.batch_size,-1))
        loss_matrix.append(loss.view(1,-1))

    loss_matrix = torch.cat(loss_matrix, dim=0)

    masked = loss_matrix * mask[1:]#TODO
    lossPerDay = torch.sum(masked, dim = 1)/torch.sum(mask[1:], dim=1 ) #1*daylen
    loss = torch.sum(masked[:daylen]) / torch.sum(mask[1:][:daylen])



    return lossPerDay, loss.item()








def test(args, model, test_path, criterion):
    current_loss = 0
    
    lossPerDays = []
    
    lossPerDays_avg = []

    model.eval()

    daylen = args.daytolook
    with torch.no_grad():
        iloop =0
        test_iter = FSIterator(test_path, args.batch_size, 1)
        for input, target, mask in test_iter:

            lossPerDay, loss = evaluate(args, model, input, mask, target, criterion)
            lossPerDays.append(lossPerDay[:daylen]) #n_batches * 10
            current_loss += loss
            iloop+=1

        lossPerDays = torch.stack(lossPerDays)
        lossPerDays_avg = lossPerDays.sum(dim =0)


        lossPerDays_avg = lossPerDays_avg/iloop

        current_loss = current_loss/iloop

    return lossPerDays_avg, current_loss
