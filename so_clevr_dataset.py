from __future__ import print_function, division

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class SortOfClevrDataset(Dataset):
    def __init__(self, dir, filename, only_rel=False, only_norel=False,transform=None):
        path = os.path.join(dir, filename)
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        # Pre-processing
        self.rel = []
        self.norel = []
        for img, rel_qas, norel_qas in dataset:
            img = np.swapaxes(img, 0, 2)
            for qst, ans in zip(rel_qas[0], rel_qas[1]):
                self.rel.append((img, qst, ans))
            for qst, ans in zip(norel_qas[0], norel_qas[1]):
                self.norel.append((img, qst, ans))

        if only_rel:
            self.dataset = self.rel
        elif only_norel:
            self.dataset = self.norel
        else:
            self.dataset = self.rel+self.norel

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, question, answer = self.dataset[idx]

        image = np.float32(image)/255.
        image = torch.from_numpy(image)

        question = np.float32(question)
        question = torch.from_numpy(question)

        sample = {'image': image, 'question': question, 'answer': answer}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

