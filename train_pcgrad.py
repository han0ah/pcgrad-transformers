import sys
import torch
import logging
import argparse
from torch.utils.data import DataLoader
from model import RobertaForMTL
from transformers import AutoTokenizer, AdamW
from data import PAWSDataset, KlueNLIDataset
from pcgrad import PCGrad

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(model, evalloader, device, task_id):
    model.eval()
    total_eval_loss = 0.0
    pred_list = []
    label_list = []
    with torch.no_grad():
        for i, d in enumerate(evalloader):
            outputs = model(
                input_ids=d['input_ids'].to(device),
                attention_mask=d['attention_mask'].to(device),
                labels=d['labels'].to(device),
                task_id=task_id
            )
            loss, logits = outputs['loss'], outputs['logits']
            total_eval_loss += (loss.item()*d['labels'].shape[0])
            pred_list.append(torch.argmax(logits, dim=-1))
            label_list.append(d['labels'])
    
    pred_list = torch.cat(pred_list).to(device)
    label_list = torch.cat(label_list).to(device)

    total_cnt = pred_list.shape[0]
    correct_cnt = torch.sum(pred_list==label_list).item()
    return total_eval_loss/total_cnt, correct_cnt/total_cnt


def main(args):
    model = RobertaForMTL.from_pretrained(args.plm_path)
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Task 0 (PAWS) Dataset & DataLoader
    task0_train = PAWSDataset(args.paws_train_path, tokenizer, max_n=20000)
    task0_valid = PAWSDataset(args.paws_valid_path, tokenizer, max_n=20000)
    task0_trainloader = DataLoader(task0_train, batch_size=args.batch_size, shuffle=True)
    task0_validloader = DataLoader(task0_valid, batch_size=args.batch_size, shuffle=False)

    # Task 1 (KLUE-NLI) Dataset & DataLoader
    task1_train = KlueNLIDataset(args.nli_train_path , tokenizer)
    task1_valid = KlueNLIDataset(args.nli_valid_path, tokenizer)
    task1_trainloader = DataLoader(task1_train, batch_size=args.batch_size, shuffle=True)
    task1_validloader = DataLoader(task1_valid, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    optimizer = PCGrad(optimizer)

    for e_i in range(args.epoch):
        model.train()
        for i, (d0,d1) in enumerate(zip(task0_trainloader, task1_trainloader)):
            outputs0 = model(
                input_ids=d0['input_ids'].to(device),
                attention_mask=d0['attention_mask'].to(device),
                labels=d0['labels'].to(device),
                task_id=0
            )
            loss0 = outputs0[0]

            outputs1 = model(
                input_ids=d1['input_ids'].to(device),
                attention_mask=d1['attention_mask'].to(device),
                labels=d1['labels'].to(device),
                task_id=1
            )
            loss1 = outputs1[0]

            optimizer.pc_backward([loss0,loss1])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        eloss, acc = evaluate(model, task0_validloader, device, task_id=0)
        logger.info(f" [Epoch {e_i}] ---")
        logger.info("  > Task 0 (PAWS)     Eval Loss : {:.5f}, Accuracy : {:.5f}".format(eloss, acc))
        eloss, acc = evaluate(model, task1_validloader, device, task_id=1)
        logger.info("  > Task 1 (KLUE-NLI) Eval Loss : {:.5f}, Accuracy : {:.5f}".format(eloss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--plm_path', default='.../klue/roberta-base')
    parser.add_argument('--paws_train_path', default='.../dataset/paws/translated_train.tsv')
    parser.add_argument('--paws_valid_path', default='.../dataset/paws/dev_2k.tsv')
    parser.add_argument('--nli_train_path', default='.../dataset/klue-nli-v1.1/klue-nli-v1.1_train.json')
    parser.add_argument('--nli_valid_path', default='.../dataset/klue-nli-v1.1/klue-nli-v1.1_dev.json')
    args = parser.parse_args()
    main(args)