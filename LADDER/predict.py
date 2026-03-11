import numpy as np
import os
import sys
import torch
import argparse
import ladder as ladder
from dataset import SequenceFeature, GenomicFeature

def main():
    argv = argparse.ArgumentParser(description='Lamina Associated Domain finDER (LADDER) Model Prediction.')
    
    argv.add_argument('--output', dest='output', help='Prediction output path', required=True)
    
    argv.add_argument('--model', dest='model_path', help='Trained model path', required=True)
    
    argv.add_argument('--species', dest='species', help='Predicted data species', required=True)
    
    argv.add_argument('--assembly', dest='genome_assembly', help='Predicted data genome assembly', required=True)
    
    argv.add_argument('--chr', dest='chromosome', help='Chromosome to predict', required=True)
    
    argv.add_argument('--chrbins', dest='cnbins', help='Chromosome length in 200kb resolution', type=int, required=True)
    
    argv.add_argument('--seq', dest='sequence_path', help='Chromosome sequence data location', required=True)
    
    argv.add_argument('--genedensity', dest='genedensity_path', help='Chromosome gene density data location', required=True)
    
    argv.add_argument('--linedensity', dest='linedensity_path', help='Chromosome line data location', required=True)
    
    argv.add_argument('--sinedensity', dest='sinedensity_path', help='Chromosome sine data location', required=True)
    
    args_ = argv.parse_args(args=None if sys.argv[1:] else ['--help'])
    
    prediction_(args_.output, args_.species, args_.genome_assembly, args_.chromosome, args_.cnbins, args_.model_path, args_.sequence_path, args_.genedensity_path, args_.linedensity_path, args_.sinedensity_path)

def prediction_(output_p, species, genome_assembly, chrname, nbins, model_p, seq_p, genedensity_p, linedensity_p, sinedensity_p):
    for nb in range(nbins):
        start = nb * 200000
        feature_p = [seq_p, genedensity_p, linedensity_p, sinedensity_p]
        feature_inp = dataloader(chrname, start, feature_p)
        model = get_model(model_p)
        pred = model(feature_inp)[0].detach().cpu().numpy()
        os.makedirs('%s/npy/%s/%s' % (output_p, species, genome_assembly), exist_ok = True)
        np.save('%s/npy/%s/%s/%s_%d' % (output_p, species, genome_assembly, chrname, start), pred)
        
def dataloader(chrname, start, feature_p, width = 524288):
    end = start + width
    f_r = []
    for f_p in feature_p:
        if "dna" in f_p:
            f = SequenceFeature(path = '%s/%s.fa.gz' % (f_p, chrname))
            f_r.append(torch.tensor(f.get(start, end)).unsqueeze(0))
        else:
            f = GenomicFeature(path = f_p, norm = 'log')
            f_r.append(torch.tensor(np.nan_to_num(f.get(chrname, start, end),0)))
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
