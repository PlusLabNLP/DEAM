import amrlib
import numpy as np
import argparse
from tqdm import tqdm

#loading amr-text and text-amr models
stog = amrlib.load_stog_model(model_dir='amr_models/model_parse_t5-v0_1_0', device='cuda:0')
gtos = amrlib.load_gtos_model('amr_models/model_generate_t5-v0_1_0', device='cuda:0')


def load_specific_conv_amrs(conv):
        '''load the specified conversation's amrs
        Param:
            conv: a specific conversation
        '''
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
        parser.add_argument('--fname', type=str, default='train', help='type of data to generate amr parsing for its convs')
        parser.add_argument('--path', type=str, default='data/topical_chat/', help='the path of the file')	
        args = parser.parse_args()
        fname='{}/{}.txt'.format(args.path, args.fname)
        foutput='{}/parsed_{}.txt'.format(args.path, args.fname)

        fr = open(fname, 'r')
        convs = fr.readlines()
        filter_convid=False
        fw=open(foutput, 'w')
        if 'topical_chat' in args.path:
            filter_convid=True
        for conv in tqdm(convs):
                if filter_convid:
                    conv=conv.split('</CONV>')[1]
                utts=conv.split('</UTT>')
                conv_amrs = stog.parse_sents(utts)
                include=True
                for c in conv_amrs:
                    if not c:
                         include=False
                         continue
                if include:
                     conv_amrs = [c.replace('\n', '\\n') for c in conv_amrs]
                     conv_amrs = '</UTT>'.join(conv_amrs)
                     fw.write(conv_amrs+'\n')
                
               

        

