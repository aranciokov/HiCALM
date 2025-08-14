# HiCALM
Implementation of HiCALM, Hierarchical Cross-modal Alignment of Language and Metaverse. It extends our previous work presented at [MMM 2025](https://mmm2025.net/), [HierArtEx](https://link.springer.com/chapter/10.1007/978-981-96-2061-6_5) ([pdf preprint](https://ailab.uniud.it/wp-content/uploads/2024/12/MMM2025_HierArtEx.pdf)).

## Brief Description

Metaverse environments provide immersive, multimedia-rich experiences with increasing importance in education, entertainment, and cultural contexts. Understanding these environments, often composed of multiple interconnected rooms or subspaces, is crucial for building effective Metaverse retrieval systems. Yet, existing methods fall short, as they are not designed to disentangle local semantics (e.g., individual multimedia elements, room-level details) from global, Metaverse-wide meaning. We present HiCALM, a hierarchical Metaverse retrieval framework that mirrors the structural hierarchy of Metaverse environments. HiCALM models them in a bottom-up manner, progressively capturing both local and global semantics. To bridge the gap between visual and textual modalities, we introduce a cross-modal hierarchical loss that trains the model to align hierarchical visual features with hierarchically extracted textual information, enabling more accurate text-to-Metaverse retrieval. Extensive quantitative and qualitative experiments demonstrate that HiCALM delivers substantial improvements over existing approaches, achieving up to 95.0% R@1 and 75.7% nDCG@5 (+47.3% and +16.4%, respectively).

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

To run the evaluation code (also computes nDCG, which is not computed at training time due to higher cost):

```
python eval_main_HierArtEx_v2_.py --room_vis_agg rnn --room_txt_agg rnn --generalist mobile_clip
```

## Data

Domain-specific features (folders starting with "preextracted_vectors_wikiart_") and CLIP features (folder "tmp_museums/open_clip_features_museums3k") can be copied from https://github.com/aranciokov/HierArtEx-MMM2025

Moreover, the following symbolic link should be created:
```
a=$PWD
ln -s $a/tmp_museums/museums3k_new_features/mobile_clip/descriptions/tokens_strings /tmp_museums/museums3k_new_features/blip_base/descriptions/tokens_strings
```

In the end, this folder will have this structure:
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

## Details

Machine setup: python 3.10.12 (CUDA 11.7), single A5000 GPU, Intel Xeon W-2123 CPU (3.60GHz)

Main libraries used: torch 1.13.1, numpy 1.23.5, pandas 1.5.3


## Citation

\[journal version soon\]

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
