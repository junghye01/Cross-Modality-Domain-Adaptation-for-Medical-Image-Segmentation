import os
from light_training.dataloading.dataset import get_train_val_loader_from_train
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric_modified import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses import DiceCELoss  # Import for multiclass segmentation loss
from monai.metrics import DiceMetric
import numpy as np
import wandb
import argparse
import sys

set_determinism(123)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['USE_SIMPLE_THREADED_LEVEL3'] = '1'

logdir = f"./logs/segmamba"
model_save_path = os.path.join(logdir, "model")
augmentation = True

env = "pytorch"
num_gpus = 1
device = "cuda:0"
class_names={1:'VS', 2:'Cochlea'}

roi_size = [128, 128, 128]

#class_weights=[0,0.07788523,0.92211477] #  0.07788523 0.92211477

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, lr, model_name, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.5)
        self.augmentation = augmentation
        from model_segmamba.segmamba import SegMamba

        # Assuming `out_chans` is the number of classes
        self.num_classes = 3 # include background class  # Specify the number of classes for multi-class segmentation
        self.model = SegMamba(in_chans=1, out_chans=self.num_classes, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384])

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.model_name = model_name

        # Use DiceCELoss for multi-class segmentation

       # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
       # print(f' length of class weights : {len(class_weights_tensor)}')

       # self.loss_fn = DiceCELoss(include_background=False,to_onehot_y=True, softmax=True,weight=class_weights_tensor)

        #self.loss_fn=nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_fn=DiceCELoss(include_background=False,to_onehot_y=True, softmax=True)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"

        # Initialize Dice Metric for each class
        #self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)


    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)
        
        print(f'pred shape: {pred.shape}, label shape: {label.shape}')
        #print(f'label_mask shape :{label_mask.cpu().detach().numpy().shape}, pred shape :{pred.cpu().detach().numpy().shape}') 1,1, 128,128,128 1,3,128,128,128
        loss = self.loss_fn(pred, label)
       
        wandb.log({'loss': loss})
        self.log("training_loss", loss, step=self.global_step)
        return loss 


    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label[label==-1]=0

        #label = nn.functional.one_hot(label.squeeze(1).long(), num_classes=self.num_classes)
        #label = label.permute(0, 4, 1, 2, 3)  # Change shape to (B, C, H, W, D)
        return image, label

    def cal_metric(self, gt, pred):
       
        
        gt = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt
        pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
       # print(f'gt shape : {gt.shape}, pred shape: {pred.shape}') # pred_labels의 shape이 2면 잘못된 거임 
        pred_labels=np.argmax(pred,axis=1) # 각 행을 따라.. 
        gt = gt.squeeze(1)  # Remove channel dimension if necessary

        #print(f'gt shape : {gt.shape}, pred_labels shape: {pred_labels.shape}') # pred_labels의 shape이 2면 잘못된 거임 
        print(f'gt unique values : {np.unique(gt)}, pred labels unique values : {np.unique(pred_labels)}')
       
       
        dice_scores = []
        for class_idx in range(1,self.num_classes):
            dice_score = dice(test=(pred_labels == class_idx).astype(int), reference=(gt == class_idx).astype(int))
            dice_scores.append(dice_score)

        return np.array(dice_scores)

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.model(image)
        
        output=torch.softmax(output,dim=1)
        #ce
        
        dice_scores = self.cal_metric(label, output)
        return dice_scores  # Return numpy array with dice scores for each class

    def validation_end(self, val_outputs):
        # Convert val_outputs to a numpy array
        val_outputs = np.array(val_outputs)
        
        mean_dice_per_class=np.nanmean(val_outputs,axis=0)
        
        mean_dice=np.nanmean(mean_dice_per_class)

        print(f'val_outputs shape : {val_outputs.shape}')
        print(f"Mean Dice per class: {mean_dice_per_class}")
        print(f"Overall Mean Dice: {mean_dice}")

        self.log("mean_dice", mean_dice, step=self.epoch)
        for class_idx, class_dice in enumerate(mean_dice_per_class):
            self.log(f"mean_dice_class_{class_names[class_idx+1]}", class_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, os.path.join(model_save_path, f"best_{self.model_name}_{mean_dice:.4f}.pt"), delete_symbol=f"best_{self.model_name}_")

        save_new_model_and_delete_last(self.model, os.path.join(model_save_path, f"final_{self.model_name}_{mean_dice:.4f}.pt"), delete_symbol=f"final_{self.model_name}_")

        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(), os.path.join(model_save_path, f"tmp_model_3_ep{self.epoch}_{mean_dice:.4f}.pt"))

        # wandb
        wandb.log({'mean_dice': mean_dice})
        for class_idx, class_dice in enumerate(mean_dice_per_class):
            wandb.log({f'mean_dice_class_{class_names[class_idx+1]}': class_dice})

if __name__ == "__main__":
    wandb.login(key='7cb4cc9dd9130889291245c37464f5799921ee84')
    wandb.init(project='Multiclass_SegMamba_Segmentation')

    parser = argparse.ArgumentParser(description='train SegMamba')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_every', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model_name', type=str, default='v2', help='save model name')
    parser.add_argument('--data_dir', type=str, default='../data/preprocessed_data/Tr')
    args = parser.parse_args()

    trainer = BraTSTrainer(env_type=env, max_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, model_name=args.model_name, device=device, logdir=logdir, val_every=args.val_every, num_gpus=num_gpus, master_port=17759, training_script=__file__)
    train_ds, val_ds = get_train_val_loader_from_train(args.data_dir)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

