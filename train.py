from __future__ import print_function, division

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.base_model import net as base_model
from models.improved_model import net as improved_model
from models.improved_model2 import net as improved_model2

from so_clevr_dataset import SortOfClevrDataset

from util.train_test_fn import train, test
from util.torch_util import save_checkpoint, str_to_bool, print_info

if __name__ == '__main__':
    print_info('[Relational Reasoning Network] training script',['green','bold'])

    # Argument parsing
    parser = argparse.ArgumentParser(description='Relational Reasoning Network PyTorch Implementation')
    # Paths
    parser.add_argument('--trained-models-dir', type=str, default='trained_models', help='path to trained models folder')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
    # Model parameters
    parser.add_argument('--model-type', type=str, default='base', choices=['base', 'improved', 'improved2'], help='Model type: base, improved, improved2')
    parser.add_argument('--load-model', type=str, default='', help='loading the trained model checkpoint')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    # Seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # RN model and loss
    print('Creating RN model...')
    if args.model_type == 'base':
        model = base_model(question_len=11, n_feature=256, n_classes=10, use_cuda=use_cuda)
    elif args.model_type == 'improved':
        model = improved_model(question_len=11, n_feature=256, n_classes=10, use_cuda=use_cuda)
    elif args.model_type == 'improved2':
        model = improved_model2(question_len=11, n_feature=256, n_classes=10, use_cuda=use_cuda)

    print('Using NLL loss...')
    loss = nn.NLLLoss()

    # Dataset and dataloader
    dataset_train = SortOfClevrDataset(dir='datasets', filename='sort-of-clevr_train.pickle')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    dataset_test = SortOfClevrDataset(dir='datasets', filename='sort-of-clevr_test.pickle')
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # The number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Train
    best_test_loss = float("inf")

    print('Starting training...\n')
    print_info("# ============================= #\n"
               " ...Train config...\n"
               " ------------------\n"
               " * # of train data: "+str(len(dataset_train))+"\n\n"
               " * Learning rate: "+str(args.lr)+"\n"
               " * Batch size: "+str(args.batch_size)+"\n"
               " * Maximum epoch: "+str(args.num_epochs)+"\n\n"
                                                           
               " * # of parameters: "+str(total_params)+"\n"
               "# ============================= #\n",['yellow','bold'])

    start_epoch = 1
    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        # Load model state dict
        model.load_state_dict(checkpoint['state_dict'])
        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Load epoch information
        start_epoch = checkpoint['epoch']
        print("Reloading from--[%s]" % args.load_model)

    for epoch in range(start_epoch, args.num_epochs+1):
        # Call train, test function
        train_loss = train(epoch,model,loss,optimizer,dataloader_train,use_cuda,log_interval=100)
        test_acc = test(model,dataloader_test,len(dataset_test),use_cuda)

        checkpoint_name = os.path.join(args.trained_models_dir, args.model_type+'_epoch_'+str(epoch)+'.pth.tar')

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },checkpoint_name)

    print('Done!')
