# LWSR: Lifelong Histopathology Whole Slide Image Retrieval via Distance Consistency Rehearsal

[MICCAI 2024] Lifelong Histopathology Whole Slide Image Retrieval via Distance Consistency Rehearsal.

## Data preparation

#### - Download slides and convert into tiles 

You may need to download the slides from TCGA and convert them into tiles before using the scripts in this repository.

Download code please refer to [https://github.com/MarvinLer/tcga_segmentation](https://github.com/MarvinLer/tcga_segmentation)

Convert slides into tiles code please refer to [https://github.com/Zhengyushan/tcga2tile](https://github.com/Zhengyushan/tcga2tile)

#### - Extract feature from tiles

Run the codes to extract feature from tiles on a single GPU:

```shell
python -u extract_feature/extract_feature.py 
		--dataset path_of_dataset_id_file/TCGA-BRCA.txt
		--batch-size 512 
		--device cuda:0 
		--output-path path_to_save_feature/TCGA-BRCA/ 
```

Please refer to [PLIP](https://github.com/PathologyFoundation/plip) for weight checkpoint downloading and detailed usage.
#### - Structure of the whole slide image dataset to run the code.

```
./data                                                              	# The directory of the feature data.
├─ TCGA-BRCA     														# The directory for a cancer subtype.
│  ├─ TCGA-4H-AAAK-01Z-00-DX1.ABF1B042-1970-4E28-8671-43AAD393D2F9.pth
|  ├─ TCGA-5L-AAT0-01Z-00-DX1.5E171263-30BF-4C6B-88A1-E8EA0522A861.pth
|  ├─ TCGA-5L-AAT1-01Z-00-DX1.F3449A5B-2AC4-4ED7-BF44-4C8946CDB47D.pth
|  └─ ... 
│     
├─ TCGA-COAD     														# The directory for a cancer subtype.
│  ├─ TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.pth
|  ├─ TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.pth
|  ├─ TCGA-4N-A93T-01Z-00-DX1.82E240B1-22C3-46E3-891F-0DCE35C43F8B.pth
|  └─ ... 
│     
├─ TCGA-ESCA     														# The directory for a cancer subtype.
│  ├─ TCGA-2H-A9GF-01Z-00-DX1.FA1016AF-3FE3-45DC-A77B-F1ACC2B33B2A.pth
|  ├─ TCGA-2H-A9GG-01Z-00-DX1.0C979026-128C-4124-96CB-0B93FF35CFFF.pth
|  ├─ TCGA-2H-A9GH-01Z-00-DX1.B2BF80D6-D348-4C5F-A205-6827684BF3B6.pth
|  └─ ... 
│     
```
## Train
Run the codes on a single GPU:
```shell
python -u utils/main.py 
		--device cuda:0 
		--sampling_num 2048 
		--pair_loss_weight 1.0 
		--ce_loss_weight 1.0 
		--dc_loss_weight 0.01 
		--model lwsr 
		--lr 1e-5 
		--dataset seq_tcga 
		--exp_desc lwsr_training 
		--buffer_size 100 
		--n_classes 8 
		--n_epochs 70 
		--batch_size 10 
		--minibatch_size 30 
		--checkpoints_save_path path_to_save_checkpoint
```
## Reference
If the code is helpful to your research, please cite:
```
@InProceedings{Zhu_Lifelong_MICCAI2024,
	author = { Zhu, Xinyu and Jiang, Zhiguo and Wu, Kun and Shi, Jun and Zheng, Yushan},
	title = { { Lifelong Histopathology Whole Slide Image Retrieval via Distance Consistency Rehearsal } },
	booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
	year = {2024},
	publisher = {Springer Nature Switzerland},
	volume = {LNCS 15004},
	month = {October},
	page = {274 -- 284}
}
```
## Acknowledgements
Framework code for continual learning baseline and compared methods was largely adapted via making modifications to [Mammoth](https://github.com/aimagelab/mammoth), thanks to their wonderful work!
