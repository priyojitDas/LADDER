import numpy as np
import pandas as pd
import os
import sys
import torch
import argparse
import ladder as ladder
from dataset import SequenceFeature, GenomicFeature

def main():
    argv = argparse.ArgumentParser(description='Lamina Associated Domian finDER (LADDER) Model Prediction.')
    
    argv.add_argument('--output', dest='output', help='Prediction output path', required=True)
    
    argv.add_argument('--model', dest='model_path', help='Trained model path', required=True)
    
    argv.add_argument('--species', dest='species', help='Predicted data species', required=True)
    
    argv.add_argument('--assembly', dest='genome_assembly', help='Predicted data genome assembly', required=True)
    
    argv.add_argument('--chr', dest='chromosome', help='Chromosome to predict', required=True)
    
    argv.add_argument('--chrbins', dest='cnbins', help='Chromosome length in 200kb resolution', type=int, default=0, required=False)
    
    argv.add_argument('--seq', dest='sequence_path', help='Chromosome sequence data location', required=True)
    
    argv.add_argument('--genedensity', dest='genedensity_path', help='Chromosome gene density data location', required=True)
    
    argv.add_argument('--linedensity', dest='linedensity_path', help='Chromosome line data location', required=True)
    
    argv.add_argument('--sinedensity', dest='sinedensity_path', help='Chromosome sine data location', required=True)

    argv.add_argument('--location', dest='location_path', help='Genomic regions location', required=True)
    
    args_ = argv.parse_args(args=None if sys.argv[1:] else ['--help'])
    
    prediction_(args_.output, args_.species, args_.genome_assembly, args_.chromosome, args_.cnbins, args_.model_path, args_.sequence_path, args_.genedensity_path, args_.linedensity_path, args_.sinedensity_path, args_.location_path)
    prediction_up(args_.output, args_.species, args_.genome_assembly, args_.chromosome, args_.cnbins, args_.model_path, args_.sequence_path, args_.genedensity_path, args_.linedensity_path, args_.sinedensity_path, args_.location_path)
    prediction_down(args_.output, args_.species, args_.genome_assembly, args_.chromosome, args_.cnbins, args_.model_path, args_.sequence_path, args_.genedensity_path, args_.linedensity_path, args_.sinedensity_path, args_.location_path)

def prediction_(output_p, species, genome_assembly, chrname, nbins, model_p, seq_p, genedensity_p, linedensity_p, sinedensity_p, location_p):
    del_df = pd.read_csv(location_p,header=None,sep='\t')
    del_df[2] = del_df[2] - 1
    del_df = del_df[del_df[0] == chrname]
    for di in range(del_df.shape[0]):
        start = del_df.iloc[di,1] - 262144
        if start < 0:
            continue
        feature_p = [seq_p, genedensity_p, linedensity_p, sinedensity_p]
        feature_inp = dataloader(chrname, start, feature_p, [del_df.iloc[di,1],del_df.iloc[di,2]], deltype='center')
        model = get_model(model_p)
        pred = model(feature_inp)[0].detach().cpu().numpy()
        os.makedirs('%s/npy/%s/%s' % (output_p, species, genome_assembly), exist_ok = True)
        np.save('%s/npy/%s/%s/%s_%d_%s' % (output_p, species, genome_assembly, chrname, start, del_df.iloc[di,3].lower()), pred)
        
def prediction_up(output_p, species, genome_assembly, chrname, nbins, model_p, seq_p, genedensity_p, linedensity_p, sinedensity_p, location_p):
    del_df = pd.read_csv(location_p,header=None,sep='\t')
    del_df[2] = del_df[2] - 1
    del_df = del_df[del_df[0] == chrname]
    for di in range(del_df.shape[0]):
        start = del_df.iloc[di,1]
        if start < 0:
            continue
        feature_p = [seq_p, genedensity_p, linedensity_p, sinedensity_p]
        feature_inp = dataloader(chrname, start, feature_p, [del_df.iloc[di,1],del_df.iloc[di,2]], deltype='up')
        model = get_model(model_p)
        pred = model(feature_inp)[0].detach().cpu().numpy()
        os.makedirs('%sup/npy/%s/%s' % (output_p, species, genome_assembly), exist_ok = True)
        np.save('%sup/npy/%s/%s/%s_%d_%s' % (output_p, species, genome_assembly, chrname, start, del_df.iloc[di,3].lower()), pred)

def prediction_down(output_p, species, genome_assembly, chrname, nbins, model_p, seq_p, genedensity_p, linedensity_p, sinedensity_p, location_p):
    del_df = pd.read_csv(location_p,header=None,sep='\t')
    del_df[2] = del_df[2] - 1
    del_df = del_df[del_df[0] == chrname]
    for di in range(del_df.shape[0]):
        start = del_df.iloc[di,1] - 524288
        if start < 0:
            continue
        feature_p = [seq_p, genedensity_p, linedensity_p, sinedensity_p]
        feature_inp = dataloader(chrname, start, feature_p, [del_df.iloc[di,1],del_df.iloc[di,2]], deltype='down')
        model = get_model(model_p)
        pred = model(feature_inp)[0].detach().cpu().numpy()
        os.makedirs('%sdown/npy/%s/%s' % (output_p, species, genome_assembly), exist_ok = True)
        np.save('%sdown/npy/%s/%s/%s_%d_%s' % (output_p, species, genome_assembly, chrname, start, del_df.iloc[di,3].lower()), pred)

def dataloader(chrname, start, feature_p, pos_df, deltype, width = 524288):
    end = start + width + (pos_df[1] - pos_df[0] + 1)
    f_r = []
    for f_p in feature_p:
        if "dna" in f_p:
            f = SequenceFeature(path = '%s/%s.fa.gz' % (f_p, chrname))
            if deltype == 'center':
                f_r.append(torch.tensor(f.getdel(start, end, pos_df)).unsqueeze(0))
            elif deltype == 'up':
                f_r.append(torch.tensor(f.getdelup(start, end, pos_df)).unsqueeze(0))
            elif deltype == 'down':
                f_r.append(torch.tensor(f.getdeldown(start, end, pos_df)).unsqueeze(0))
        else:
            f = GenomicFeature(path = f_p, norm = 'log')
            if deltype == 'center':
                f_r.append(torch.tensor(np.nan_to_num(f.getdel(chrname, start, end, pos_df),0)))
            elif deltype == 'up':
                f_r.append(torch.tensor(np.nan_to_num(f.getdelup(chrname, start, end, pos_df),0)))
            elif deltype == 'down':
                f_r.append(torch.tensor(np.nan_to_num(f.getdeldown(chrname, start, end, pos_df),0)))
    f_seq = f_r[0]
    f_den = torch.cat([f_d.unsqueeze(0).unsqueeze(2) for f_d in f_r[1:]], dim = 2)
    inp = torch.cat([f_seq, f_den], dim = 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = inp.to(device)
    return inp

def get_model(model_p):
    model_name =  'CTGModel'
    num_genomic_features = 3
    ModelClass = getattr(ladder, model_name)
    model = ModelClass(num_genomic_features, mid_hidden = 256)
    load_checkpoint(model, model_p)
    return model

def load_checkpoint(model, model_path):
    print('Loading weights')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model_weights = checkpoint['state_dict']

    for key in list(model_weights):
        model_weights[key.replace('model.', '')] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()
    return model

if __name__ == '__main__':
    main()
