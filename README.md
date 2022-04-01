The official implementation of SIGIR 2022 paper ".

## Code

This code is mainly adapted from [KenChan's keyphrase generation code](https://github.com/kenchan0226/keyphrase-generation-rl) and [TAKG](https://github.com/yuewang-cuhk/TAKG). Thanks for their work.
### Quick Start
The whole process includes the following steps:
Preprocessing,Training,Decoding,Evaluation

The datasets can be downloaded from [here](https://drive.google.com/file/d/1pyVHkrddoI8qumJkovz1_ss4kaQKe7oS/view?usp=sharing)

#### Preprocess

You can downloaded our Preprocess data and best model [here](https://drive.google.com/file/d/1CMcI5vE9Pkfxe5g9X5Y7iTVcy_jrGO1M/view?usp=sharing)

If you downloaded our Preprocess data you can skip the Preprocess step.

You need to preprocess kp20k first,then you can get the vocab.pt and the dataset for training.

```
python preprocess-data_dir data/kp20k
```

The 4 test datasets(inspec,krapivin,nus,semeval) need to use the vocab build by propeocess kp20k,you need to use -vocab_dir parameter

```
python preprocess-data_dir data/inspec -vocab processed_data/kp20k_s512_t10
```

It will output the processed data to the folder processed_data.

#### Training
before joint train, you need to train the hierarchy ntm.

```
python train.py -data_tag kp20k_s512_t10 -only_train_ntm -ntm_warm_up_epochs 120 -batch_size 512 -topic_num 20 -n_topic2 5 -gpuid 0
```

then train the keyphrase generation model

```
python train.py -data_tag kp20k_s512_t10 -copy_attention -joint_train -check_pt_ntm_model_path [the warmed up ntm model path] -load_pretrain_ntm -save_each_epoch -batch_size 128 -topic_num 20 -n_topic2 5 -epoch 20 -gpuid 1
```

There are some common arguments about different joint training strategies and model variants:

```
-topic_num:the number of topic
-n_topic2:the number of topic tree
```

#### Decoding 

To generate the prediction, run: 

```python
python predict.py -model [seq2seq model path] -ntm_model [ntm model path]
```

Once the decoding finished, it creates a predictions.txt in the path specified by pred_path, e.g., pred/predict_kp20k_s512_t10.joint_train.copy.seed9527.topic_num20.emb150.vs50000.dec300.20211219-221957_e10.train_loss=18.455.val_loss=19.743.model-15h-28m/predictions.txt For each line in the prediction.txt contains all the predicted keyphrases for a source.

For some reason if want to decode the 4 test datasets, you need to copy the model path and rename the dataset

```
model/kp20k_s512_t10.joint_train.copy.seed9527.topic_num20.emb150.vs50000.dec300.20211219-221957/e10.train_loss=18.455.val_loss=19.743.model-15h-28m
```

to

```
model/inspec(krapivin,nus,semeval)_s512_t10.joint_train.copy.seed9527.topic_num20.emb150.vs50000.dec300.20211219-221957/e10.train_loss=18.455.val_loss=19.743.model-15h-28m
```

for 4 datasets you need to do the same operation 4 times.

#### Evaluation

To run evaluate: 

```
python pred_evaluate.py-pred_file_path [path to pred file] -exp_path test -src_file_path [path to src file] -trg_file_path [path to trg file] -disable_extra_one_word_filter -invalidate_unk -export_filtered_pred -all_ks O 5 10 -present_ks O 5 10 -absent_ks O 5 10
```

#### Result

you can downloaded the raw predictions and corresponding evaluation results [here](https://drive.google.com/file/d/1-4pL9nVNAgKQpYrNVxeERkyHKTwGFnNa/view?usp=sharing),which contains the following files:

```

├── MyPred
│   ├── inspec_s512_t10.txt
│   ├── kp20k_s512_t10.txt
│   ├── krapivin_s512_t10.txt
│   ├── nus_s512_t10.txt
│   └── semeval_s512_t10.txt
├── MyResult
│   ├── pred result
```

