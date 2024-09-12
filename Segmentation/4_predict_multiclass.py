import numpy as np
from light_training.dataloading.dataset import get_test_loader_from_test
import torch
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric_modified import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.prediction import Predictor
import os
import argparse

env = "pytorch"
num_gpus = 1
device = "cuda:0"
roi_size = [128, 128, 128]

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, model_path, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.5)
        self.model_path = model_path
        self.augmentation = False
        self.num_classes = 3  # For multi-class segmentation

    def define_model_segmamba(self):
        from model_segmamba.segmamba import SegMamba
        model = SegMamba(in_chans=1, out_chans=self.num_classes, depths=[2,2,2,2], feat_size=[48, 96, 192, 384])
        
        # Load the trained model
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.to(device)
        model.eval()

        predictor = Predictor(window_infer=self.window_infer)

        save_path = "./prediction_results/Ts_cm_tr"
        os.makedirs(save_path, exist_ok=True)

        return model, predictor, save_path

    def get_input(self, batch):
        print(f'batch keys : {batch.keys()}')
        image = batch["data"]
        label = batch.get("seg", None)
        #print(f'Loaded label : {label}')
        properties = batch.get("properties", None)

        for k,v in properties.items():
            if v is None:
                print(f'key {k} property value is None')
                
        return image, label, properties

    def cal_metric(self, gt, pred):
        # Convert tensors to numpy arrays for Dice calculation
        gt = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt
        pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred

        # Convert to predicted class labels
        pred_labels = np.argmax(pred, axis=1)
        
        #print(f'pred shape :{pred.shape} pred_labels shape :{pred_labels.shape}. gt shape :{gt.shape}')
        
        gt = gt.squeeze(1)  # Remove channel dimension if necessary
        
        # Calculate Dice score for each class
        dice_scores = []
        for class_idx in range(1,self.num_classes):
            #if class_idx==0:
           #     continue
            dice_score = dice(test=(pred_labels == class_idx).astype(int), reference=(gt == class_idx).astype(int))
            
            dice_scores.append(dice_score)

        return np.array(dice_scores)

    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        
        model, predictor, save_path = self.define_model_segmamba()

        with torch.no_grad():
            model_output = predictor.maybe_mirror_and_predict(image, model, device=device)
            model_output = torch.softmax(model_output, dim=1).cpu().numpy()  # Softmax for multi-class probabilities

        # Calculate Dice score for each sample
        if label is not None:
            dice_scores=self.cal_metric(label,model_output)
           # dice_scores=np.array(dice_scores)
            #print(f'dice_scores shape :{dice_scores.shape}')
            mean_dice_per_class=dice_scores
            mean_dice=np.nanmean(mean_dice_per_class)
        
            #dice_scores_per_class = []
            #for i in range(len(model_output)):
            #    dice_scores = self.cal_metric(label[i], model_output[i])
            #    dice_scores_per_class.append(dice_scores)

            #dice_scores_per_class = np.array(dice_scores_per_class)
            #mean_dice_per_class = np.nanmean(dice_scores_per_class, axis=0)
            #overall_mean_dice = np.nanmean(mean_dice_per_class)
            #print(f"Mean Dice per class: {mean_dice_per_class}")
            #for class_idx,class_dice in enumerate(mean_dice_per_class):
            #    print(f'mean_dice_class_{class_idx}: {class_dice}')
            #print(f"Overall Mean Dice: {mean_dice}")
        else:
            print('Labels are not provided, skipping metric calculation.')
            mean_dice = 0.0
            mean_dice_per_class=np.zeros(self.num_classes-1)

        # Optionally, save the output
        for i, case_name in enumerate(properties['name']):
            predictor.save_to_nii(model_output[i], image,raw_spacing=[1,1,1], case_name=case_name, save_dir=save_path)

        return mean_dice_per_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SegMamba')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_every', type=int, default=2)
    parser.add_argument('--model_path', type=str, default='./logs/segmamba/model/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='../data/preprocessed_data/Ts')
    args = parser.parse_args()

    trainer = BraTSTrainer(env_type=env, max_epochs=args.epochs, batch_size=args.batch_size, model_path=args.model_path, device=device, logdir="", val_every=args.val_every, num_gpus=num_gpus, master_port=17751, training_script=__file__)
    
    test_ds = get_test_loader_from_test(args.data_dir)
    trainer.validation_single_gpu(test_ds)
