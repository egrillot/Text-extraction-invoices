import argparse
from transformers.models.layoutlm import LayoutLMTokenizer
from layoutlm.data.funsd import FunsdDataset
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn import CrossEntropyLoss
from transformers import LayoutLMForTokenClassification
import torch
from tqdm import tqdm
import numpy as np
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,dest='dp')
parser.add_argument('--model_path',type=str,dest='mp')
args = parser.parse_args()

dir_path=args.dp

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels
labels = get_labels(dir_path+'labels.txt')
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}
pad_token_label_id = CrossEntropyLoss().ignore_index

args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': dir_path,
        'model_name_or_path':'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlmv2',}
# class to turn the keys of a dict into attributes
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
args = AttrDict(args)
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

eval_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="test")
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset,
                             sampler=eval_sampler,
                            batch_size=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels)
model.load_state_dict(torch.load(args.mp, map_location=device))
model.to(device)
# put model in evaluation mode
model=model.eval()

eval_loss = 0.0
nb_eval_steps = 0
preds = None
out_label_ids = None

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    with torch.no_grad():
        input_ids = batch[0].to(device)
        bbox = batch[4].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)
# forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=labels)
        # get the loss and logits
        tmp_eval_loss = outputs.loss
        logits = outputs.logits
eval_loss += tmp_eval_loss.item()
nb_eval_steps += 1
# compute the predictions
if preds is None:
    preds = logits.detach().cpu().numpy()
    out_label_ids = labels.detach().cpu().numpy()
else:
    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    out_label_ids = np.append(
    out_label_ids, labels.detach().cpu().numpy(), axis=0
            )
# compute average evaluation loss
eval_loss = eval_loss / nb_eval_steps
preds = np.argmax(preds, axis=2)
out_label_list = [[] for _ in range(out_label_ids.shape[0])]
preds_list = [[] for _ in range(out_label_ids.shape[0])]
for i in range(out_label_ids.shape[0]):
    for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
            out_label_list[i].append(label_map[out_label_ids[i][j]])
            preds_list[i].append(label_map[preds[i][j]])
results = {
    "loss": eval_loss,
    "precision": precision_score(out_label_list, preds_list),
    "recall": recall_score(out_label_list, preds_list),
    "f1": f1_score(out_label_list, preds_list),
}

print(results)
