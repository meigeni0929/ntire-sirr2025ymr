import os
from os.path import join

import torch.backends.cudnn as cudnn

# import data.sirs_dataset as datasets
import data.dataset_sir as datasets
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils

opt = TrainOptions().parse()

opt.isTrain = False   
cudnn.benchmark = True
opt.no_log = True     
opt.display_id = 0    
opt.verbose = False   
datadir = os.path.join(os.path.expanduser('~'), './Alldata')
#real
eval_dataset_real = datasets.DSRTestDataset(join(datadir, '/test1'), if_align=opt.if_align)
#val
eval_dataset_solidobject = datasets.DSRTestDataset(join(datadir, '/test2'),if_align=opt.if_align)
#train
eval_dataset_postcard = datasets.DSRTestDataset(join(datadir, '/test3'), if_align=opt.if_align)
eval_dataset_wild = datasets.DSRTestDataset(join(datadir, '/test4'), if_align=opt.if_align)
#加载四个数据集
eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=True,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_wild = datasets.DataLoader(
    eval_dataset_wild, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

engine = Engine(opt, eval_dataset_real)

"""Main Loop"""
result_dir = os.path.join('./Alldata', opt.name, mutils.get_formatted_time())
#评估结果
res1 = engine.eval(eval_dataloader_real, dataset_name='test1',
                  savedir=join(result_dir, 'testout'), suffix='testout')

#res2 = engine.eval(eval_dataloader_solidobject, dataset_name='test2',
#                  savedir=join(result_dir, 'test2'), suffix='test2')
#res3 = engine.eval(eval_dataloader_postcard, dataset_name='train',
#                  savedir=join(result_dir, 'train'), suffix='train')

#res4 = engine.eval(eval_dataloader_wild, dataset_name='train2',
#                  savedir=join(result_dir, 'train2'), suffix='train2')

