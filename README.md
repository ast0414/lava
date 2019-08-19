# LAVA
This repository contains the source code for the paper:
*An, Sungtae, et al. "Longitudinal Adversarial Attack on Electronic Health Records Data." Proceedings of the 2019 World Wide Web Conference, WWW'19, ACM, 2019.* [Paper](http://delivery.acm.org/10.1145/3320000/3313528/p2558-an.html?ip=143.215.34.117&id=3313528&acc=ACTIVE%20SERVICE&key=A79D83B43E50B5B8%2E5E2401E94B5C98E0%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1563333251_e85a371b7110fd4ef78589d9af78835f)

##### These codes were tested with Python 3.6 and PyTorch 0.3.1 (built with CUDA 9.1). You may use the included conda environment file `environment.yml`.

## 0. Preprocess Data
### 0.0. Prepare MIMIC-III Dataset
While we used a different private EHR data for the experiments reported in the paper, here we use [MIMIC-III](https://mimic.physionet.org/) as an example dataset since it is publicly available upon request. You should obtain access and download the CSV database first. Please refer to their [web page](https://mimic.physionet.org/gettingstarted/access/).

Once the MIMIC-III CSV files are ready, you can use the following command to process the data into the required format.
```bash
python 0_0_process_mimic.py <ADMISSIONS FILE PATH> <DIAGNOSES FILE PATH> <PATIENTS FILE PATH> <OUTPUT PATH WITH A FILE NAME PREFIX>
```

For example,
```bash
python 0_0_process_mimic.py /mimic3/ADMISSIONS.csv /mimic3/DIAGNOSES_ICD.csv /mimic3/PATIENTS.csv ./Processed/outout
```
Among the multiple output files, `output.seqs` and `output.morts` are mostly used in this project.
##### `output.seqs`
It is a pickled Python object, which is a List of List of List of Int.
First of all, all clinical (event) features such as diagnoses, procedures, and medications are encoded as integer values from 0 to N-1 where N is the number of clinical features. Then, each visit (or multiple events on the same date) made by a patient is grouped and represented as a List of Int. Similarly, a single patient's clinical history (prior to a certain index date determined) is represented as a List of visits (which is a List of Int above) ordered by time/date. Finally, the entire cohort (all patients) data are collected into a List. Therefore, the final form is a List of List of List of Int.

##### `output.mort`
It is a pickled Python List that contains 0 or 1 values as a label (mortality) for each patient. It has the same order of patients as `output.seqs`. 

You may use any other datasets with this LAVA code once they are prepared in the same format described above.

### 0.1. Split the dataset
We split the prepared dataset into three folds: training, validation, and test sets.
You may use your own script or the following code:
```bash
python 0_1_split_dataset.py <PICKLED SEQUENCE FILE> <PICKLED LABEL FILE> <OUTPUT DIR>
```

For example,
```bash
python 0_1_split_dataset.py Processed/output.seqs Processed/output.morts Processed/
```
It will generate `<FOLD NAME>.seqs` and `<FOLD NAME>.labels` for this LAVA project, and additional `<FOLD NAME>_<NORMALIZED>.data_csr` files which can be used as non-sequence input data, with MLP for example.

## 1. Train a source (RETAIN) model
Now that we have a pre-processed dataset, we train a source model to craft adversarial exmaples of longitudinal EHR by using LAVA. In the paper and so here, we use RETAIN as our source model to utilize dual-level attention mechanism. Please refer to [LAVA paper](http://delivery.acm.org/10.1145/3320000/3313528/p2558-an.html?ip=143.215.34.117&id=3313528&acc=ACTIVE%20SERVICE&key=A79D83B43E50B5B8%2E5E2401E94B5C98E0%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1563333251_e85a371b7110fd4ef78589d9af78835f) and the [RETAIN paper](https://papers.nips.cc/paper/6321-retain-an-interpretable-predictive-model-for-healthcare-using-reverse-time-attention-mechanism) for more details.

You can train a source RETAIN model using the following code:

```bash
python 1_train_RETAIN.py <DATASET DIR>
```
For example,

```bash
python 1_train_RETAIN.py Processed/
```

**Please note that you must set the number of available features in advance in the code or by using the argument `--num_features`. It applies same for all the codes below.**

Please refer to the code or `python 1_train_RETAIN --help` to check available options for the model architecture or the training procedure.


## 2. Craft LAVA samples
Once we have a trained source model, we can craft LAVA adversarial examples:

```bash
python 2_craft_lava.py <CLEAN DATASET DIR> <TRAINED MODEL FILE>
```
For example,

```bash
python 2_craft_lava.py Processed/ Save/best_model.pth
```
Please refer to the code or `python 2_craft_lava.py --help` to check available options for the crafting procedure.

## 3. Test a target model
We can evaluate a trained model with either the clean test set or the crafted adversarial test set.
While we reported the results for a variety of different target models in the paper, here we use the same source RETAIN model as the target model (i.e., white-box attack) for simplicity.

You can test/evaluate a trained RETAIN model using the following code:
```bash
python 3_test_RETAIN.py <PICKLED TEST SEQUENCE FILE> <PICKLED TEST LABEL FILE> <TRAINED MODEL FILE>
```

For example,
##### Clean test set

```bash
python 3_test_RETAIN.py Processed/test.seqs Processed/test.labels Save/best_model.pth
```

##### Crafted adversarial test set (note that we use the same label file)
```bash
python 3_test_RETAIN.py Crafted/adv_20.seqs Processed/test.labels Save/best_model.pth
```

Please contact **Sungtae An <stan84@gatech.edu>** if you have any questions about either the paper or the code.
