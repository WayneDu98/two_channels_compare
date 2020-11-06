#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: DuJiabao
# @Date  : 2020/10/21
# @Desc  : 

import torch


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, maxpooling=2):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(maxpooling),
            torch.nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class TwoChannelsCNN(torch.nn.Module):
    def __init__(self):
        super(TwoChannelsCNN, self).__init__()
        self.convblock1 = ConvBlock(in_channels=6, out_channels=32, kernel_size=7, stride=1, padding=3, maxpooling=3)
        self.convblock2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, maxpooling=3)
        self.convblock3 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, maxpooling=2)
        self.convblock4 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, maxpooling=2)
        self.linear1 = torch.nn.Linear(256 * 3 * 3, 64)
        self.linear2 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x.squeeze()


if __name__ == "__main__":
    img = torch.rand((4, 6, 128, 128))
    model = TwoChannelsCNN()
    pred = model(img)
    criterion = torch.nn.BCEWithLogitsLoss()
    y = torch.tensor([1., 0., 0., 1.])
    # print(pred.shape)
    print(criterion(pred, y))
