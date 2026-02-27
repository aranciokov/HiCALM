# HiCALM
Implementation of HiCALM, Hierarchical Cross-modal Alignment of Language and Metaverse, accepted at [**ACM TOMM**](https://dl.acm.org/doi/10.1145/3799427)! It extends our previous work presented at [MMM 2025](https://mmm2025.net/), [HierArtEx](https://link.springer.com/chapter/10.1007/978-981-96-2061-6_5) ([pdf preprint](https://ailab.uniud.it/wp-content/uploads/2024/12/MMM2025_HierArtEx.pdf)).

## Brief Description

Metaverse environments provide immersive, multimedia-rich experiences with increasing importance in education, entertainment, and cultural contexts. Understanding these environments, often composed of multiple interconnected rooms or subspaces, is crucial for building effective Metaverse retrieval systems. Yet, existing methods fall short, as they are not designed to disentangle local semantics (e.g., individual multimedia elements, room-level details) from global, Metaverse-wide meaning. We present HiCALM, a hierarchical Metaverse retrieval framework that mirrors the structural hierarchy of Metaverse environments. HiCALM models them in a bottom-up manner, progressively capturing both local and global semantics. To bridge the gap between visual and textual modalities, we introduce a cross-modal hierarchical loss that trains the model to align hierarchical visual features with hierarchically extracted textual information, enabling more accurate text-to-Metaverse retrieval. Extensive quantitative and qualitative experiments demonstrate that HiCALM delivers substantial improvements over existing approaches, achieving up to 95.0% R@1 and 75.7% nDCG@5 (+47.3% and +16.4%, respectively). It was also tested on a new dataset, GamingMV, of gaming-relateved metaverses, achieving again substantial improvements compared to the other methods that we tested.

## Examples

To run the code for the baseline:

```
python run_main_HierArtEx_v2_.py --generalist clip --no-hiervis --no-hiertxt --no-artexp 
```

Final method:

```
python run_main_HierArtEx_v2_.py --room_vis_agg rnn --room_txt_agg rnn --generalist mobile_clip
```

Some ablation studies (e.g., Table 1.a-b):
```
python run_main_HierArtEx_v2_.py --room_vis_agg avg --room_txt_agg avg --generalist mobile_clip ;
python run_main_HierArtEx_v2_.py --room_vis_agg monornn --room_txt_agg avg --generalist mobile_clip ;
python run_main_HierArtEx_v2_.py --room_vis_agg rnn --room_txt_agg avg --generalist mobile_clip ;
python run_main_HierArtEx_v2_.py --room_vis_agg avg --room_txt_agg monornn --generalist mobile_clip ;
python run_main_HierArtEx_v2_.py --room_vis_agg avg --room_txt_agg rnn --generalist mobile_clip ;
```

To run the evaluation code after having trained a model (also computes nDCG, which is not computed at training time due to higher cost):

```
python eval_main_HierArtEx_v2_.py --room_vis_agg rnn --room_txt_agg rnn --generalist mobile_clip
```

To run the evaluation code on the user queries (after having trained a model):

```
python eval_main_HierArtEx_v2_.py --room_vis_agg rnn --room_txt_agg rnn --generalist mobile_clip --eval-user-queries
```

Some examples of runs on GamingMV are also available in the experiments_gaming.sh file.

## Data (Museums3k)

Domain-specific features (folders starting with "preextracted_vectors_wikiart_") and CLIP features (folder "tmp_museums/open_clip_features_museums3k") can be copied from https://github.com/aranciokov/HierArtEx-MMM2025

Moreover, the following symbolic link should be created:
```
a=$PWD
ln -s $a/tmp_museums/museums3k_new_features/mobile_clip/descriptions/tokens_strings /tmp_museums/museums3k_new_features/blip_base/descriptions/tokens_strings
```

In the end, the folder will have this structure:
```
- preextracted_vectors_wikiart_rn101
- preextracted_vectors_wikiart_rn50
- ...
- preextracted_vectors_wikiart_vitb32
- tmp_museums
    - open_clip_features_museums3k
        - descriptions
        - images
    - museums3k_new_features
        - blip_base
            - descriptions
            - images
        - mobile_clip
            - descriptions
            - images
```

## Data (GamingMV)

Pre-extracted MobileCLIP and CLIP features are available in the dataset_gaming_VR folder, whereas the domain-expert features (extracted using ResNet50, pretrained on gaming data) are available in the folder gaming_DE_features. Raw frames can be obtained by downloading the videos from youtube and then extracting the frames (see https://github.com/aliabdari/GamingMV). Here, we also performed manual frame filtering (indices available in GamingMV_frames_index.json) and adjusted the pre-extracted features so that only the features of retained frames are present.

The folder should have this structure:
```
- (possibly the other folders for Museum3k dataset)
- dataset_gaming_VR
    - filtered_video_game_Features
        - ViT-B-32_laion2b_s34b_b79k
        - mobile_clip
- gaming_DE_features
```

## Details

Machine setup: python 3.10.12 (CUDA 11.7), single A5000 GPU, Intel Xeon W-2123 CPU (3.60GHz)

Main libraries used: torch 1.13.1, numpy 1.23.5, pandas 1.5.3


## Citation

```
@article{10.1145/3799427,
  author = {Abdari, Ali and Falcon, Alex and Serra, Giuseppe},
  title = {Retrieving Relevant Metaverses using Hierarchical Features},
  year = {2026},
  publisher = {Association for Computing Machinery},
  issn = {1551-6857},
  doi = {10.1145/3799427},
  note = {Just Accepted},
  journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
}
```

```
@inproceedings{falcon2024hierartex,
  title={Hierartex: hierarchical representations and art experts supporting the retrieval of museums in the metaverse},
  author={Falcon, Alex and Abdari, Ali and Serra, Giuseppe},
  booktitle={International Conference on Multimedia Modeling},
  pages={60--73},
  year={2024},
  organization={Springer}
}
```
