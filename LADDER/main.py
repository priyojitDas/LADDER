import sys
import torch
from torch.nn import functional as F
import argparse
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pl_bolts
import ladder as ladder
from dataset import GenomicDataset

def main():
    args = parser()
    start_training(args)

def parser():
  argv = argparse.ArgumentParser(description='Lamina Associated Domain finDER (LADDER) Model.')

  argv.add_argument('--output', dest='output', default='checkpoints', help='Model output path')

  argv.add_argument('--input', dest='input', default='data', help='Training data path', required=True)
  
  argv.add_argument('--assembly', dest='genome_assembly', default='hg19', help='Training data genome assembly')

  argv.add_argument('--epochs', dest='epochs', default=80, type=int, help='# of epochs')
  
  argv.add_argument('--gpu', dest='ngpus', default=4, type=int, help='# of GPUs')

  argv.add_argument('--batch-size', dest='batch_size', default=8, type=int, help='Batch size')


  args_ = argv.parse_args(args=None if sys.argv[1:] else ['--help'])
  return args_

def start_training(args):

    stop_early = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=80, verbose=False, mode="min")
    model_outs = callbacks.ModelCheckpoint(dirpath='%s/models' % (args.output), save_top_k=80, monitor='val_loss')
    learning_rate_m = callbacks.LearningRateMonitor(logging_interval='epoch')

    save_csv = pl.loggers.CSVLogger(save_dir = '%s/csv' % (args.output))
    all_loggers = save_csv
    
    pl.seed_everything(12345, workers=True)
    tr_module = TrainModule(args)
    trainer = pl.Trainer(strategy='ddp', accelerator="gpu", devices=args.ngpus, gradient_clip_val=1, logger = all_loggers,callbacks = [stop_early, model_outs, learning_rate_m], max_epochs = args.epochs)
    train_dataloader = tr_module.get_dataloader(args, 'train')
    val_dataloader = tr_module.get_dataloader(args, 'val')
    trainer.fit(tr_module, train_dataloader, val_dataloader)

class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.model = self.get_model()
        self.args = args
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        seq, features, lad, start, end, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        lad = lad.float()
        return inputs, lad
    
    def training_step(self, batch, batch_idx):
        inputs, lad = self.proc_batch(batch)
        outputs = self(inputs)
        loss = F.mse_loss(outputs, lad)
        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, batch_size = inputs.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, lad = self.proc_batch(batch)
        outputs = self(inputs)
        loss = F.mse_loss(outputs, lad)
        return loss

    def on_training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        mean_loss = torch.tensor(step_outputs).mean()
        metrics = {'train_loss' : mean_loss}
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        mean_loss = torch.tensor(step_outputs).mean()
        metrics = {'val_loss' : mean_loss}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = 2e-5,
                                     weight_decay = 0)

        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.args.epochs)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'WarmupCosineAnnealing',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}

    def get_dataset(self, args, mode):

        datatype_root = '%s/%s' % (args.input, args.genome_assembly)
        genomic_features = {'genedensity' : {'file_name' : 'genedensity.bw',
                                             'norm' : 'log' },
                            'linedensity' : {'file_name' : 'linedensity.bw',
                                             'norm' : 'log' },
                            'sinedensity' : {'file_name' : 'sinedensity.bw',
                                             'norm' : 'log' }
                            }
        dataset = GenomicDataset(datatype_root, 
                                args.genome_assembly,
                                genomic_features, 
                                mode = mode)
        
        if mode == 'val':
            self.val_length = len(dataset) / args.batch_size
            print('Validation loader length:', self.val_length)

        return dataset

    def get_dataloader(self, args, mode):
        dataset = self.get_dataset(args, mode)

        if mode == 'train':
            shuffle = True
        else:
            shuffle = False

        gpus = args.ngpus
        batch_size = int(args.batch_size / gpus)
        num_workers = int(16 / gpus) 

        dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, prefetch_factor=1, persistent_workers=True)
        return dataloader

    def get_model(self):
        model_name =  'CTGModel'
        num_genomic_features = 3
        ModelClass = getattr(ladder, model_name)
        model = ModelClass(num_genomic_features, mid_hidden = 256)
        return model

if __name__ == '__main__':
    main()
