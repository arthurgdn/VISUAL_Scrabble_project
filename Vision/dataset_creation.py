import torch
import numpy as np
import cv2
from datasets import Dataset


def compute_metrics(p, metric):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def create_dataset(list_img, label = 8):
    """creates a dataset from the list of image (list_img). 
    This dataser will help fine tuning the model"""
    out = {'image':[], 'labels':[]}
    for image in list_img:
        out['image'].append(image)
        out['labels'].append(label)
    return Dataset.from_dict(out)

# def process_example(example, extractor):
#     inputs = extractor(example['image'], return_tensors='pt')
#     inputs['labels'] = example['labels']
#     return inputs

def transform(example_batch, extractor):
    # Take a list of PIL images and turn them to pixel values
    inputs = extractor([cv2.cvtColor(np.float32(np.array(x)), cv2.COLOR_GRAY2BGR) for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

