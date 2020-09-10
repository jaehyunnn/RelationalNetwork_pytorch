from __future__ import print_function, division

import argparse
import torch
from torch.utils.data import DataLoader

from models.base_model import net as base_model
from models.improved_model import net as improved_model
from models.improved_model2 import net as improved_model2

from so_clevr_dataset import SortOfClevrDataset

from util.torch_util import BatchTensorToVars, str_to_bool, print_info
from tqdm import tqdm

if __name__ == '__main__':
    print_info('[Relational Reasoning Network] Evaluation Script', ['green', 'bold'])

    # Argument parser
    parser = argparse.ArgumentParser(description='Relational Reasoning Network PyTorch Implementation')

    parser.add_argument('--model-type', type=str, default='base', choices=['base', 'improved', 'improved2'], help='Model type: base, improved, improved2')
    parser.add_argument('--load-model', type=str, default='trained_models/base_model.pth.tar', help='The trained model path')
    parser.add_argument('--batch-size', type=int, default=64, help='Test batch size')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    # Create model
    print('Creating RN model...')
    if args.model_type == 'base':
        model = base_model(question_len=11, n_feature=256, n_classes=10, use_cuda=use_cuda)
    elif args.model_type == 'improved':
        model = improved_model(question_len=11, n_feature=256, n_classes=10, use_cuda=use_cuda)
    elif args.model_type == 'improved2':
        model = improved_model2(question_len=11, n_feature=256, n_classes=10, use_cuda=use_cuda)

    # Load trained weights
    print('Loading trained model weights...')
    checkpoint = torch.load(args.load_model, map_location=lambda storage, loc: storage)

    # Load model state dict
    model.load_state_dict(checkpoint['state_dict'])

    # Dataset and dataloader
    dataset_rel = SortOfClevrDataset(dir='datasets', filename='sort-of-clevr_test.pickle', only_rel=True)
    dataset_norel = SortOfClevrDataset(dir='datasets', filename='sort-of-clevr_test.pickle', only_norel=True)

    if use_cuda:
        batch_size = args.batch_size
    else:
        batch_size = 1

    dataloader_rel = DataLoader(dataset_rel, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_norel = DataLoader(dataset_norel, batch_size=batch_size, shuffle=False, num_workers=4)

    batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)

    print('Computing Accuracy...')
    print("[%s]"%args.load_model)
    print_info("# ============================= #\n"
               " ...Eval config...\n"
               " ------------------\n"
               " * # of eval data: " + str(len(dataset_rel)+len(dataset_norel)) + "\n"
               " * Batch size: " + str(args.batch_size) + "\n"
               "# ============================= #\n", ['yellow', 'bold'])

    total_correct_rel = 0
    total_correct_norel = 0

    # Inference relation dataset
    for i, batch in enumerate(tqdm(dataloader_rel)):
        batch = batchTensorToVars(batch)
        image, question, answer = batch['image'], batch['question'], batch['answer']
        if use_cuda:
            image = image.cuda()
            question = question.cuda()
            answer = answer.cuda()

        model.eval()
        output = model(image, question)

        pred = output.data.max(1)[1]
        correct = pred.eq(answer.data).cpu().sum()
        total_correct_rel += correct
    accuracy_rel = total_correct_rel * 100. / len(dataset_rel)

    # Inference non-relation dataset
    for i, batch in enumerate(tqdm(dataloader_norel)):
        batch = batchTensorToVars(batch)
        image, question, answer = batch['image'], batch['question'], batch['answer']
        if use_cuda:
            image = image.cuda()
            question = question.cuda()
            answer = answer.cuda()

        model.eval()
        output = model(image, question)

        pred = output.data.max(1)[1]
        correct = pred.eq(answer.data).cpu().sum()
        total_correct_norel += correct
    accuracy_norel = total_correct_norel * 100. / len(dataset_norel)

    accuracy_total = (total_correct_rel+total_correct_norel)*100. / (len(dataset_rel)+len(dataset_norel))
    # Print results
    print('')
    print('# ======== Accuracy ======== #')
    print(' * Non-relation : %.1f ' % accuracy_norel)
    print(' * Relation : %.1f ' % accuracy_rel)
    print()
    print(' * Total : %.1f ' % accuracy_total)
    print('# ========================== #')

