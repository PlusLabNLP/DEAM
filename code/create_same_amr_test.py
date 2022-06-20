import amrlib
import numpy as np
import argparse
from tqdm import tqdm

stog = amrlib.load_stog_model(model_dir='model_parse_t5-v0_1_0', device='cuda:0')
gtos = amrlib.load_gtos_model('model_generate_t5-v0_1_0', device='cuda:0')

np.random.seed(1000)

def load_specific_conv_amrs(conv):
        utts=conv.split('</UTT>')[:-1]
        label=conv.split('</UTT>')[-1]
        amrs = stog.parse_sents(utts)
        utts_amrs=[]
        utts=[]
       
        for ind, amr in enumerate(amrs):
                parts = amr.split('\\n')
                utts_amrs.append('\n'.join(parts[1:]))
                utts.append(parts[0].split('# ::snt')[1].strip())
        return utts_amrs, label

if __name__=='__main__':
        parser = argparse.ArgumentParser(description='create convs from same amr')
        parser.add_argument('--fname', type=str, default='train', help='type of data to generate convs from its amrs')
        parser.add_argument('--path', type=str, default='data', help='the path of the file')	
        args = parser.parse_args()
        fname='{}/{}.txt'.format(args.path, args.fname)
        foutput='{}/{}_same.txt'.format(args.path, args.fname)

        fr = open(fname, 'r')
        convs = fr.readlines()
        fw=open(foutput, 'w')
        
        for conv in tqdm(convs):
                utts=conv.split('</UTT>')[:-1]
                label=conv.split('</UTT>')[-1]
                conv_amrs = stog.parse_sents(utts)
                sents=[]
                if None in conv_amrs:
                    sents_amr, _ = gtos.generate([c for c in conv_amrs if c != None])
                    i=0
                    for ind, utt in enumerate(conv_amrs):
                        if utt==None:
                            sents.append(utts[ind])
                            #print('if utt amr is None, we copy the same sentence {}'.format(utts[ind]))
                        else:
                            sents.append(sents_amr[i])
                            i+=1
                else:
                    sents, _ = gtos.generate(conv_amrs)
                same_conv = '</UTT>'.join(sents) + '</UTT>{}'.format(label)
                fw.write(same_conv)
                #print(conv)
                #print(same_conv)
                

        

