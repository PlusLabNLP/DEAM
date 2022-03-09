#DEAM

This repository contains code for [DEAM]() paper. If you use it please cite it as: 
```
@inproceedings{ghazarian2022deam,
  title={DEAM: Dialogue Coherence Evaluation using AMR-based Semantic Manipulations},
  author={Ghazarian, Sarik and Wen, Nuan and Galstyan, Aram and Peng, Nanyun},
  booktitle={Proceedings of the Conference of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2022}
}
```

For any comments/issues/ideas pleaae feel free to contact [me](mailto:sarikgha@usc.edu).


## Install Requirements
Use any virtualenv manager to install all the packages mentioned in the requirements.txt file.

In order to train DEAM, first you need to create negative samples using AMRs.

##Steps for AMR-based manipulations 

### 1. Parse conversations and get their AMRs
In this step, first you need to have text-to-AMR and AMR-to-text models. 
Follow [amrlib](https://github.com/bjascob/amrlib) github page to download two finetuned T5 models 1) model_parse_t5-v0_1_0 as the text-to-AMR model and model_generate_t5-v0_1_0 as the AMR-to-text model and save them in a folder called "amr_models". We use the first model to parse the conversations and retrieve their AMRs. 

python utils/amr_parse_convs.py --path data/topical_persona/ --fname train



### 2. AMRs manipulations
In this step, we apply DEAM's manipulations to the extracted AMRs of the conversations and then generate conversations from the manipulated AMRs leveraging the AMR-to-text model. 

python code/manipulate_amr_incoherent_convs_submit_ver.py --data_path data/topical_persona/ --o_data_path data/topical_persona/ --fname train



### 3. Train DEAM
Next, we finetune the Roberta-Large model on the set of data which includes both coherent/incoherent conversations resulted from AMRs/Manipulated\_AMRs.

python code/ft_coh_model.py \
        --mode train \
        --train_data_path data/topical_persona/train_amr_manamr_cont_coref_pirel_eng.txt \
        --valid_data_path data/topical_persona/valid_amr_manamr_cont_coref_pirel_eng.txt \
        --model_path  coh_models/ \
        --model_name roberta-large --model_type roberta \



### 4. Predict DEAM
We use DEAM to predict scores for [FED](http://shikib.com/fed_data.json) and [DSTC9](https://github.com/exe1023/DialEvalMetrics/tree/main/data/dstc9_data) test sets.

python code/ft_coh_model.py \
        --mode predict \
        --train_data_path data/topical_persona/train_amr_manamr_cont_coref_pirel_eng.txt \
        --valid_data_path data/topical_persona/valid_amr_manamr_cont_coref_pirel_eng.txt \
        --model_path  coh_models/ \
        --model_name roberta-large --model_type roberta \




### 5. Compute Correlation
At the end, we compute the correlation of scores predicted by DEAM for the test sets with human judgments.

python utils/compute_correlation.py  --model_path coh_models/



##Ablation Study
We ignore each of the four manipulations in the negative samples generation step to examine their effectiveness. The results of the ablated files can be found in the data/topical_persona/ directory.
To train DEAM using ablated data, predict scores and compute the correlation between DEAM's scores with human judgmeents, you can follow next steps (here we only show the scripts for the Contradiction ablation as an example):


python3 code/ft_coh_model.py \
        --mode train \
        --train_data_path data/topical_persona/train_amr_manamr_ablated_cont.txt \
        --valid_data_path data/topical_persona/valid_amr_manamr_ablated_cont.txt \
        --model_path  coh_models/ablation_cont/\
        --model_name roberta-large --model_type roberta \
        --learning_rate=2e-6 \
        --max_length=512 \
        --train_batch_size=2 --valid_batch_size=2 \
        --logging_steps=600 --save_steps=500


python3 code/ft_coh_model.py \
        --mode predict \
        --train_data_path data/topical_persona/train_amr_manamr_ablated_cont.txt \
        --valid_data_path data/topical_persona/valid_amr_manamr_ablated_cont.txt \
        --model_path  coh_models/ablation_cont/\
        --model_name roberta-large --model_type roberta \
   

python utils/compute_correlation.py  --model_path coh_models/ablation_cont/






