# import matplotlib.pyplot as plt
import os  # contains utilities for reading directories on disk
import re

import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.use_deterministic_algorithms(True)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

# DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     worker_init_fn=seed_worker,
#     generator=g,
# )

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*10, "Using", device)

import pandas as pd
""" indices = pd.read_pickle("indices_museum_dataset.pkl")
indices['train'][:10], indices['val'][:10], indices['test'][:10], """

from torch.utils.data import Dataset


class DescriptionSceneMuseum(Dataset):
    def __init__(self, data_description_path, data_raw_description_path, data_scene_path, data_art_path, indices, split, num_imgs=96):
        self.description_path = data_description_path
        self.raw_description_path = data_raw_description_path
        self.data_pov_path = data_scene_path
        self.indices = indices[split]
        self.split = split

        available_data = [im[:-3] for im in os.listdir(data_scene_path)]
        available_data = sorted(available_data)
        available_data = [available_data[ix] for ix in self.indices.tolist()]

        self.descs = [torch.load(os.path.join(data_description_path, f"{sm}.pt"), weights_only=True) for sm in available_data]
        self.raw_descs = [" ".join(pd.read_pickle(os.path.join(data_raw_description_path, f"{sm}.pkl"))) for sm in available_data]
        self.room_desc_indices = [[
                sent_ix for sent_ix, sent in enumerate(sm.split("."))
                if re.match(r"^ In the \w+ video", sent)
        ] for sm in self.raw_descs]
        if num_imgs < 96:
            # the things we read are lists (length: Ri, num of rooms in museum i) of tensors ((N, F) num frames x features)
            # we want to uniformly sample num_imgs from each room, so that we build a tensor and move on!
            def uniform_sample(tensor, num_imgs):
                R = len(tensor)
                assert R > 0, f"tensor {tensor} has no rooms??"
                new_tensor = []
                for ten in tensor:
                    N, F = ten.shape
                    num = min(num_imgs, N)
                    idx = torch.linspace(0, N - 1, steps=num).long()
                    new_tensor.append(ten[idx])
                new_tensor = torch.stack(new_tensor)  # (R, N, F)
                new_tensor = new_tensor.view(new_tensor.shape[0] * new_tensor.shape[1], new_tensor.shape[2])  # (R*N, F)
                return new_tensor

            self.pov_images = [
                uniform_sample(
                    torch.load(os.path.join(data_scene_path, f"{sm}.pt"), weights_only=True),
                    num_imgs
                )
                for sm in available_data
            ]

            self.art_vectors = [
                uniform_sample(
                    torch.load(os.path.join(data_art_path, f"{sm}.pt"), weights_only=True),
                    num_imgs
                )
                for sm in available_data
            ]

        else:
            assert False, "not implemented: we have ragged vectors (diff number of frames per room)"
            self.pov_images = [torch.load(os.path.join(data_scene_path, f"{sm}.pt"), weights_only=True) for sm in available_data]
            self.art_vectors = [torch.load(os.path.join(data_art_path, f"{sm}.pt"), weights_only=True) for sm in available_data]
        self.names = available_data
        print(f"'{split.upper()}': {len(self.names)} names, "
              f"{len(self.descs)} sentences ({sum([len(x) for x in self.descs]) / len(self.descs)} avg "
              f" (tokens: {sum([len(x.split()) for x in self.raw_descs]) / len(self.raw_descs)} avg;"
              f" unique tokens: {len(set(' '.join(self.raw_descs).split()))} total) "
              f"{len(self.pov_images)} images ({sum([len(x) for x in self.pov_images]) / len(self.pov_images)} avg).")
        print(f"Total amount of tokens: {sum([len(x.split()) for x in self.raw_descs])}")
        """print(f"Example of raw description:")
        print(self.raw_descs[0])
        print(self.raw_descs[1])"""

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        desc_tensor = self.descs[index]
        #if self.split == "train":
        raw_desc = self.raw_descs[index]
        scene_img_tensor = self.pov_images[index]
        scene_art_tensor = self.art_vectors[index]
        #scene_art_tensor = torch.zeros((1,1))  # dummy
        name = self.names[index]
        room_desc_indices = self.room_desc_indices[index]

        #print(f"Sample #{index} ({name}): {desc_tensor.shape} desc, {scene_img_tensor.shape} scene, {scene_art_tensor.shape} art, {len(raw_desc.split()) if self.split == 'train' else 'N/A'} tokens in raw description, room_desc_indices {room_desc_indices}")
        #if self.split == "train":
        #    return desc_tensor, scene_img_tensor, scene_art_tensor, room_desc_indices, raw_desc, name, index
        #else:
        return desc_tensor, scene_img_tensor, scene_art_tensor, room_desc_indices, raw_desc, name, index
        
visual_bb_ftsize_k = {'rn18': 512, 'rn34': 512, 'rn50': 2048, 'rn101': 2048, 'vitb16': 768, 'vitb32': 768, 'openclip': 512}

GENERALIST_FEAT_SIZE_DICT = {
    'clip': 512,
    'siglip': 768,
    # 'blip': 512,
    'blip_base': 768,
    'ViT-L-14_laion2b_s32b_b82k': 768,
    'mobile_clip': 512
}

""" for vn in visual_bb_ftsize_k.keys():
    print(vn, torch.load(f"preextracted_vectors_wikiart_{vn}/Museum1554-7.unity.pt", weights_only=True).shape) """

print("="*20)

if False:
  for vn in GENERALIST_FEAT_SIZE_DICT.keys():
    print("="*10, vn)
    if vn == "clip":
        print(torch.load(f"./dataset_gaming_VR/video_game_features/ViT-B-32_laion2b_s34b_b79k/images/3t_PvlmUpyY.pt", weights_only=True).shape, torch.load(f"./dataset_gaming_VR/video_game_features/ViT-B-32_laion2b_s34b_b79k/images/3t_PvlmUpyY.pt", weights_only=True).dtype)
        print(torch.load(f"./dataset_gaming_VR/video_game_features/ViT-B-32_laion2b_s34b_b79k/descriptions/sentences/3t_PvlmUpyY.pt", weights_only=True).shape, torch.load(f"./dataset_gaming_VR/video_game_features/ViT-B-32_laion2b_s34b_b79k/descriptions/sentences/3t_PvlmUpyY.pt", weights_only=True).dtype)
        print(len(pd.read_pickle(f"./dataset_gaming_VR/video_game_features/ViT-B-32_laion2b_s34b_b79k/descriptions/tokens_strings/3t_PvlmUpyY.pkl")))
    else:
        try:
            print(torch.load(f"./dataset_gaming_VR/video_game_features/{vn}/images/3t_PvlmUpyY.pt", weights_only=True).shape, torch.load(f"./dataset_gaming_VR/video_game_features/{vn}/images/3t_PvlmUpyY.pt", weights_only=True).dtype)
            print(torch.load(f"./dataset_gaming_VR/video_game_features/{vn}/descriptions/sentences/3t_PvlmUpyY.pt", weights_only=True).shape, torch.load(f"./dataset_gaming_VR/video_game_features/{vn}/descriptions/sentences/3t_PvlmUpyY.pt", weights_only=True).dtype)
            print(len(pd.read_pickle(f"./dataset_gaming_VR/video_game_features/{vn}/descriptions/tokens_strings/3t_PvlmUpyY.pkl")))
        except:
            print("Error loading", vn)
        


# desc, scene, art, raw_desc, room_desc_indices, name, ix = train_dataset[1]
# print(f"The sample #{ix} ({name}) has a description of {len(desc)} sentences (shaped as {desc.shape} matrix), whereas there are {len(scene)} images (shaped as {scene.shape} matrix)")
# print(f"Example of raw description (capped at 100 characters): {raw_desc[:100]}")

# all_descs = [rd for rd in train_dataset.raw_descs] + [rd for rd in val_dataset.raw_descs] + [rd for rd in test_dataset.raw_descs] 
# print("tot", len(all_descs))

# n_tokens = [len(rd.split()) for rd in all_descs]
# print("avg tokens per museum", sum(n_tokens) / len(n_tokens))
# print("num tokens", sum(n_tokens))

import re
# _tmp = [rd.split(".") for rd in all_descs]
# _tmp = [[t for t in ts if "there are" in t and "painting" in t] for ts in _tmp]
# _tmp = [[re.sub(r"In the \w+ room , there are", "", t).strip() for t in ts] for ts in _tmp]
# _tmp = [[t for t in ts if len(t.split()) < 3] for ts in _tmp]
# _tmp = [[t.replace(" paintings", "") for t in ts] for ts in _tmp]
# _mm = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
# _tmp = [[_mm[t] for t in ts] for ts in _tmp]
# _x = [sum(t) for t in _tmp]
# sum(_x) / len(_x)

def cosine_sim(im, s):
    '''cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim


def create_rank(result, entire_descriptor, desired_output_index):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    position = torch.where(sorted_indices == desired_output_index)
    return position[0].item(), sorted_indices


def evaluate(output_description, output_scene, section, out_values=False, excel_format=False):
    avg_rank_scene = 0
    ranks_scene = []
    avg_rank_description = 0
    ranks_description = []

    ndcg_10_list = []
    ndcg_entire_list = []

    for j, i in enumerate(output_scene):
        rank, sorted_list = create_rank(i, output_description, j)
        avg_rank_scene += rank
        ranks_scene.append(rank)

    for j, i in enumerate(output_description):
        rank, sorted_list = create_rank(i, output_scene, j)
        avg_rank_description += rank
        ranks_description.append(rank)

    ranks_scene = np.array(ranks_scene)
    ranks_description = np.array(ranks_description)

    n_q = len(output_scene)
    sd_r1 = 100 * len(np.where(ranks_scene < 1)[0]) / n_q
    sd_r5 = 100 * len(np.where(ranks_scene < 5)[0]) / n_q
    sd_r10 = 100 * len(np.where(ranks_scene < 10)[0]) / n_q
    sd_medr = np.median(ranks_scene) + 1
    sd_meanr = ranks_scene.mean() + 1

    n_q = len(output_description)
    ds_r1 = 100 * len(np.where(ranks_description < 1)[0]) / n_q
    ds_r5 = 100 * len(np.where(ranks_description < 5)[0]) / n_q
    ds_r10 = 100 * len(np.where(ranks_description < 10)[0]) / n_q
    ds_medr = np.median(ranks_description) + 1
    ds_meanr = ranks_description.mean() + 1

    ds_out, sc_out = "", ""
    for mn, mv in [["R@1", ds_r1],
                   ["R@5", ds_r5],
                   ["R@10", ds_r10],
                   ["median rank", ds_medr],
                   ["mean rank", ds_meanr],
                   ]:
        ds_out += f"{mn}: {mv:.4f}   "

    for mn, mv in [("R@1", sd_r1),
                   ("R@5", sd_r5),
                   ("R@10", sd_r10),
                   ("median rank", sd_medr),
                   ("mean rank", sd_meanr),
                   ]:
        sc_out += f"{mn}: {mv:.4f}   "

    if out_values:
        print(section + " data: ")
        print("Scenes ranking: " + ds_out)
        print("Descriptions ranking: " + sc_out)
    if section == "test" and len(ndcg_10_list) > 0:
        avg_ndcg_10_entire = 100 * sum(ndcg_10_list) / len(ndcg_10_list)
        avg_ndcg_entire = 100 * sum(ndcg_entire_list) / len(ndcg_entire_list)
    else:
        avg_ndcg_10_entire = -1
        avg_ndcg_entire = -1
    
    if excel_format:
        print("-"*5)
        print("{ds_r1};{ds_r5};{ds_r10};{sd_r1};{sd_r5};{sd_r10};{ds_medr};{sd_medr}")
        print(f"{ds_r1},{ds_r5},{ds_r10},{sd_r1},{sd_r5},{sd_r10},{ds_medr},{sd_medr}")
        print("-"*5)
        formatted_string = f"{ds_r1},{ds_r5},{ds_r10},{sd_r1},{sd_r5},{sd_r10},{ds_medr},{sd_medr}"
        return ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, avg_ndcg_10_entire, avg_ndcg_entire, ds_medr, sd_medr, formatted_string        
    
    return ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, avg_ndcg_10_entire, avg_ndcg_entire, ds_medr, sd_medr


class LossContrastive:
    def __init__(self, name, patience=15, delta=.001, verbose=True):
        self.train_losses = []
        self.validation_losses = []
        self.name = name
        self.counter_patience = 0
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.verbose = verbose

    def on_epoch_end(self, loss, train=True):
        if train:
            self.train_losses.append(loss)
        else:
            self.validation_losses.append(loss)

    def get_loss_trend(self):
        return self.train_losses, self.validation_losses

    def calculate_loss(self, pairwise_distances, margin=.25, margin_tensor=None):
        batch_size = pairwise_distances.shape[0]
        diag = pairwise_distances.diag().view(batch_size, 1)
        pos_masks = torch.eye(batch_size).bool().to(pairwise_distances.device)
        d1 = diag.expand_as(pairwise_distances)
        if margin_tensor is not None:
            margin_tensor = margin_tensor.to(pairwise_distances.device)
            cost_s = (margin_tensor + pairwise_distances - d1).clamp(min=0)
        else:
            cost_s = (margin + pairwise_distances - d1).clamp(min=0)
        cost_s = cost_s.masked_fill(pos_masks, 0)
        cost_s = cost_s / (batch_size * (batch_size - 1))
        cost_s = cost_s.sum()

        d2 = diag.t().expand_as(pairwise_distances)
        if margin_tensor is not None:
            margin_tensor = margin_tensor.to(pairwise_distances.device)
            cost_d = (margin_tensor + pairwise_distances - d2).clamp(min=0)
        else:
            cost_d = (margin + pairwise_distances - d2).clamp(min=0)
        cost_d = cost_d.masked_fill(pos_masks, 0)
        cost_d = cost_d / (batch_size * (batch_size - 1))
        cost_d = cost_d.sum()

        return (cost_s + cost_d) / 2

    def is_val_improving(self):
        score = -self.validation_losses[-1] if self.validation_losses else None

        if score and self.best_score and self.verbose:
            print('epoch:', len(self.validation_losses), ' score:', -score, ' best_score:', -self.best_score, ' counter:', self.counter_patience)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter_patience += 1
            if self.counter_patience >= self.patience:
                return False
        else:
            self.best_score = score
            self.counter_patience = 0
        return True

    def save_plots(self):
        save_path = f'models/{self.name}.png'
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.validation_losses, label='Val Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Trend')

        plt.legend()

        plt.savefig(save_path)
        
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
def collate_fn(data):  # data -> desc_tensor, scene_img_tensor, name, index
    raw_descs = False
    adj = 0
    if len(data[0]) == 7:  # train -> raw descriptions
        raw_descs = True
        adj = 1

    tmp_description_povs = [x[0] for x in data] if data[0][0].dtype==torch.float16 else [x[0].to(torch.float16) for x in data]
    desc_room_indices = [x[3] for x in data]
    max_num_sents = max([max([ix_e - ix_s for ix_s, ix_e in zip([0] + dri, dri + [len(desc)])])
                         for desc, dri in zip(tmp_description_povs, desc_room_indices)])
    assert len(desc_room_indices) == len(tmp_description_povs), f"idxs {len(desc_room_indices)} desc {len(tmp_description_povs)}"
    # ======= NB right now, it will be "1st sentence before 1st room" then ["per-room paragraph"] =======
    inner_lengths = [[ix_e - ix_s for ix_s, ix_e in zip([0] + dri, dri + [len(desc)])]
                     for desc, dri in zip(tmp_description_povs, desc_room_indices)]
    # ^ this list helps us getting the final state processed by GRU to compute per-room embeddings
    
    # try:
    #     tmp_description_povs_ = []
    #     for desc, dri in zip(tmp_description_povs, desc_room_indices):
    #         tmp_description_povs_.append(pad_sequence([desc[ix_s:ix_e] for ix_s, ix_e in zip([0] + dri, dri + [-1])] + [torch.zeros(max_num_sents, 1, desc.shape[-1])],
    #                                      batch_first=True)[:-1])
    #     tmp_description_povs = tmp_description_povs_
    # except:
    #     print(max_num_sents)
    #     print(torch.zeros(max_num_sents, 1, desc.shape[-1]).shape)
    #     ln = len(tmp_description_povs)
    #     desc, dri = tmp_description_povs[ln-1], desc_room_indices[ln-1]
    #     print([desc[ix_s:ix_e].shape for ix_s, ix_e in zip([0] + dri, dri + [-1])])
    #     print(pad_sequence([desc[ix_s:ix_e] for ix_s, ix_e in zip([0] + dri, dri + [-1])],
    #                                      batch_first=True).shape)
    #     print(pad_sequence([desc[ix_s:ix_e] for ix_s, ix_e in zip([0] + dri, dri + [-1])] + [torch.zeros(max_num_sents, 1, desc.shape[-1])],
    #                                      batch_first=True).shape)
    #     print(pad_sequence([desc[ix_s:ix_e] for ix_s, ix_e in zip([0] + dri, dri + [-1])] + [torch.zeros(max_num_sents, 1, desc.shape[-1])],
    #                                      batch_first=True)[:-1].shape)
    #     assert False
        
    tmp_description_povs = [pad_sequence([desc[ix_s:ix_e] for ix_s, ix_e in zip([0] + dri, dri + [-1])] + [torch.zeros(max_num_sents, desc.shape[-1])],
                                         batch_first=True)[:-1]
                            for desc, dri in zip(tmp_description_povs, desc_room_indices)]
    
    # ^ here we have a list (len B) of padded sequences of shape ([6-9], max_num_sents, 512)
    # print(np.unique([p.shape[0] for p in tmp_description_povs]), np.unique([p.shape[1] for p in tmp_description_povs]))
    outer_lengths = [len(dri) for dri in inner_lengths]
    # ^ then we need to know where to cut to obtain again per-museum stuff
    inner_lengths = [ x for xs in inner_lengths for x in xs ]
    tmp_description_povs = [ x for xs in tmp_description_povs for x in xs ]
    # ^ here we want to concat all the padded sequences ([NNN], max_num_sents, 512)
    # =======
    tmp = pad_sequence(tmp_description_povs, batch_first=True)
    descs_pov = pack_padded_sequence(tmp,
                                     torch.tensor(inner_lengths),
                                     batch_first=True,
                                     enforce_sorted=False)

    tmp_pov = [x[1] for x in data]
    len_pov = torch.tensor([len(x) for x in tmp_pov])
    padded_pov = pad_sequence(tmp_pov, batch_first=True)
    padded_pov = torch.transpose(padded_pov, 1, 2)

    tmp_art = [x[2] for x in data]
    len_art = torch.tensor([len(x) for x in tmp_art])
    padded_art = pad_sequence(tmp_art, batch_first=True)
    padded_art = torch.transpose(padded_art, 1, 2)

    if raw_descs:
        raw_descs = [x[4] for x in data]
    names = [x[4+adj] for x in data]
    indexes = [x[5+adj] for x in data]
    
    if raw_descs:
        return descs_pov, padded_pov, padded_art, raw_descs, names, indexes, len_pov, inner_lengths, outer_lengths
    else:
        return descs_pov, padded_pov, padded_art, names, indexes, len_pov, inner_lengths, outer_lengths

def save_best_model(model_name, run_folder, *args):
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, run_folder), exist_ok=True)
    model_path = os.path.join(model_path, run_folder, model_name + '.pt')
    new_dict = dict()
    for i, bm in enumerate(args):
        new_dict[f'best_model_{str(i)}'] = bm
    torch.save(new_dict, model_path)


def load_best_model(model_name, run_folder):
    model_path = os.path.join("models", run_folder)
    avail_models = [m for m in os.listdir(model_path) if m.startswith(model_name)]
    if len(avail_models) == 1:
        model_name_ = avail_models[0]
    elif f"{model_name}.pt" in avail_models:
        model_name_ = f"{model_name}.pt"
    else:
        raise ValueError(f"Multiple models found starting with the name {model_name} in {model_path}: {avail_models}")
    model_path = model_path + os.sep + model_name_
    check_point = torch.load(model_path, weights_only=True)
    bm_list = [check_point[bm] for bm in check_point.keys()]
    return bm_list
import torch.nn as nn

class HierGRUNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False, room_txt_agg="rnn"):
        super().__init__()
        self.gru_room = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True, bidirectional=is_bidirectional)
        if room_txt_agg in ['rnn', 'monornn']:
            self.gru_museum = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=is_bidirectional)
        else:
            self.lin_museum = nn.Linear(hidden_size, hidden_size)
        self.is_bidirectional = is_bidirectional
        self.room_txt_agg = room_txt_agg

    def forward(self, x, inner_lens, outer_lens):
        # x: (B*, N, F)
        # inner_lens: [B, n_rooms,] -> number of sentences paired to each "room" (*) in each museum.
        # outer_lens: [B,] -> number of rooms per museum (+ empty/initial room/sentence).
        x = x.to(torch.float32)
        _, h_n_room = self.gru_room(x)  # (d=2, B*, F) if bidirectional else (B*, F)
        if self.is_bidirectional:
            h_n_room = h_n_room.mean(0)
        else:
            h_n_room = h_n_room.squeeze(0)
        aa = [(ol_s, ol_e) 
              for ol_s, ol_e in zip(torch.cumsum(torch.tensor([0] + outer_lens), dim=0).tolist(),
                                    torch.cumsum(torch.tensor(outer_lens + [len(outer_lens)]), dim=0).tolist())]  # -> (B, N, F)
        out_h_n_room = torch.cat([h_n_room[ol_s+1:ol_e] for ol_s, ol_e in aa], 0)  # -> (B, N, F)
        # ^ to align the room-level text vectors to the image ones, we ought to ignore the first sentence (In the museum there are...)
        x1 = [h_n_room[ol_s:ol_e] for ol_s, ol_e in aa]  # -> (B, N, F)
        if self.room_txt_agg in ['rnn', 'monornn']:
            x1 = pad_sequence(x1, batch_first=True)
            x1 = pack_padded_sequence(x1,
                                      torch.tensor(outer_lens),
                                      batch_first=True,
                                      enforce_sorted=False)
            _, h_n_museum = self.gru_museum(x1)
            if self.is_bidirectional:
                return out_h_n_room, h_n_museum.mean(0)
            else:
                return out_h_n_room, h_n_museum.squeeze(0)
        else:
            # print([a.shape for a in x1])
            x1 = pad_sequence(x1, batch_first=True)[:-1]
            h_n_museum = x1.sum(-2) / torch.tensor(outer_lens, device=x1.device).unsqueeze(1)
        return out_h_n_room.squeeze(0), h_n_museum.squeeze(0)
    
# __m=HierGRUNet(256, 256)
# __x=torch.randn((4, 6, 256))
# __il=torch.randint(3, 6, (4, )).tolist()
# __ol=torch.randint(3, 6, (4, )).tolist()
# print(__il, sum(__il), __ol, sum(__ol))
# __m(__x, __il, __ol), 


class MyHierBaseline_v3(nn.Module):
    def __init__(self, in_channels, out_channels, feature_size, art_features_size=2048, bidirectional=False, room_vis_agg="avg"):
        super().__init__()
        if art_features_size == 0:
            self.use_art = False
            self.trf_photo = nn.Linear(in_channels, out_channels)
        else:
            self.use_art = True
            self.trf_photo = nn.Linear(in_channels+art_features_size, out_channels)
        self.trf_room = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        if room_vis_agg in ['rnn', 'monornn']:
            self.trf_museum = nn.GRU(input_size=out_channels, hidden_size=feature_size, batch_first=True, bidirectional=bidirectional)
        else:
            self.trf_museum = nn.Linear(out_channels, feature_size)
        self.bidirectional = bidirectional
        self.room_vis_agg = room_vis_agg

    def forward(self, x, x_art, list_length=None, clip_mask=None, imgs_per_room=None):
        x = x.to(torch.float32)
        
        if self.use_art:
            x1 = self.trf_photo(torch.cat((
                x.transpose(1, 2), 
                x_art.transpose(1, 2)
            ), -1))
        else:
            x1 = self.trf_photo(x.transpose(1, 2))

        if clip_mask is not None:
            x1 = x1 * clip_mask
        # remove the effect of the padding
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, list_length[item_idx]:, :] = 0
        x1_img = self.relu(x1)
        
        bsz, max_n_imgs, ft_size = x1_img.shape
        if isinstance(imgs_per_room, int):
            n_rooms = max_n_imgs // imgs_per_room
            x1_room = x1_img.view(bsz, n_rooms, imgs_per_room, ft_size)
            
            # we aggregate the "image-level" information, and learn room-level information
            x1_room = x1_room.mean(2)
            x1_room = self.trf_room(x1_room)
            x1_room = self.relu(x1_room)
            
            # then, we aggregate the room-level info, and learn a museum-level representation
            assert list_length is not None
            x1_museum = x1_room.clone()
            for item_idx in range(x.shape[0]):
                x1_museum[item_idx, list_length[item_idx]:, :] = 0
                
            if self.room_vis_agg in ['rnn', 'monornn']:
            
                list_length_t = torch.tensor(list_length) if isinstance(list_length, list) else list_length
                list_length_t = (list_length_t / imgs_per_room).view(-1)
                x1_museum = pack_padded_sequence(x1_museum,
                                                 list_length_t,
                                                 batch_first=True,
                                                 enforce_sorted=False)
                _, x1_museum = self.trf_museum(x1_museum)
                x1_museum = x1_museum.mean(0) if self.bidirectional else x1_museum.squeeze(0)
            
            else:
                
                list_length_t = torch.tensor(list_length, device=x1_room.device) if isinstance(list_length, list) else list_length.to(x1_room.device)
                x1_museum = x1_museum.sum(1) / (list_length_t / imgs_per_room).view(-1, 1)
                x1_museum = self.trf_museum(x1_museum)
        
        else:
            assert False, imgs_per_room
            x1 = x1.sum(1) / (x1.sum(-1) > 0).sum(1).unsqueeze(-1)
            x1 = x1.view(x1.size(0), -1)
            x1 = self.trf_museum(x1)
        return x1_img, x1_room, x1_museum
    

    
    
############ ablations
def collate_fn_desc(data):  # data -> desc_tensor, scene_img_tensor, name, index
    raw_descs = False
    adj = 0
    if len(data[0]) == 7:  # train -> raw descriptions
        raw_descs = True
        adj = 1

    tmp_description_povs = [x[0] for x in data]
    tmp = pad_sequence(tmp_description_povs, batch_first=True)
    descs_pov = pack_padded_sequence(tmp,
                                     torch.tensor([len(x) for x in tmp_description_povs]),
                                     batch_first=True,
                                     enforce_sorted=False)

    tmp_pov = [x[1] for x in data]
    len_pov = torch.tensor([len(x) for x in tmp_pov])
    padded_pov = pad_sequence(tmp_pov, batch_first=True)
    padded_pov = torch.transpose(padded_pov, 1, 2)

    tmp_art = [x[2] for x in data]
    len_art = torch.tensor([len(x) for x in tmp_art])
    padded_art = pad_sequence(tmp_art, batch_first=True)
    padded_art = torch.transpose(padded_art, 1, 2)

    if raw_descs:
        raw_descs = [x[4] for x in data]
    names = [x[4+adj] for x in data]
    indexes = [x[5+adj] for x in data]
    
    if raw_descs:
        return descs_pov, padded_pov, padded_art, raw_descs, names, indexes, len_pov
    else:
        return descs_pov, padded_pov, padded_art, names, indexes, len_pov
    
    
class MyBaseline(nn.Module):
    def __init__(self, in_channels, out_channels, feature_size, art_features_size=0):
        super().__init__()
        if art_features_size == 0:
            self.use_art = False
            self.trf = nn.Linear(in_channels, out_channels)
        else:
            self.use_art = True
            self.trf = nn.Linear(in_channels+art_features_size, out_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, x_art=None, list_length=None, clip_mask=None):
        x = x.to(torch.float32)
        
        if self.use_art:
            assert x_art is not None
            x1 = self.trf(torch.cat((
                x.transpose(1, 2), 
                x_art.transpose(1, 2)
            ), -1))
            
        else:
            x1 = self.trf(x.transpose(1, 2))
        
        if clip_mask is not None:
            x1 = x1 * clip_mask
        # remove the effect of the padding
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, list_length[item_idx]:, :] = 0
        x1 = self.relu(x1)
        
        x1 = x1.sum(1) / (x1.sum(-1) > 0).sum(1).unsqueeze(-1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        return x1


class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True,
                          bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        x = x.to(torch.float32)
        _, h_n = self.gru(x)
        if self.is_bidirectional:
            return h_n.mean(0)
        return h_n.squeeze(0)
############ ablations
    

    
from collections import Counter

import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

def get_iou(gt, pred):
    gt_concs = gt.split(", ")
    pred_concs = pred.split(", ")
    iou = len(set(gt_concs).intersection(set(pred_concs))) / len(set(gt_concs).union(set(pred_concs)))
    return iou

def get_relevance_values(query, ranking_list):
    return [get_iou(query, museum_concepts) for museum_concepts in ranking_list]

def dcg(relevance_scores, exp=True, verbose=False):
    if exp:
        num = np.exp(relevance_scores) - 1
    else:
        num = relevance_scores
    det = np.log(np.arange(2, len(relevance_scores)+1) + 1)
    if verbose:
        print("dcg num", num, "det", det)
    return num[0] + np.sum(num[1:] / det)

def idcg(relevance_scores, exp=True, verbose=False, k=None):
    sorted_rels = sorted(relevance_scores, reverse=True)
    if k is not None:
        sorted_rels = sorted_rels[:k]
    if verbose:
        if k is None:
            print("idcg computed on", sorted_rels[:10], "...")
        else:
            print("idcg computed on", sorted_rels[:k])
    return dcg(sorted_rels, exp)

def ndcg(relevance_scores, gt_relevance_scores, exp=True, verbose=False):
    return dcg(relevance_scores, exp, verbose=verbose) / idcg(gt_relevance_scores, exp, k=len(relevance_scores), verbose=verbose)
    
bad_words =  [
    '1533', 'painting', 'paintings', 'one', 'two', 'titled', 'described', 'follows', 'work', 'room', 'picture', 'csontv', 'last', 'late', 'later', 'ois', 'period',
    'right', 'left', 'panels', 'painted', 'scene', 'first', '\'1\'', '1', 'figures', 'scenes', 'shows', 'style', 'school', '``', 'elements', 'form', 'found', 'seven', 'may',
    "''", 'wall', '2', 'st', 'van', 'made', 'head', 'di', 'genre', 'four', 'three', 'panel', 'also', 'artist', 'new', 'de', 'commissioned', 'frame', 'half', 'image',
    'pictures', 'self', 'considered', 'view', 'grand', 'main', 'painter', 'rer', 'end', 'art', 'signed', 'subject', 'figure', 'figures', 'first', 'second', 'following',
    'side', 'whose', 'seen', 'known', '000', '3', 'acts', 'age', 'almost', 'another', 'although', 'appears', 'around', 'artists', 'background', 'behind', 'brothers', 'c', 
    'centre', 'ceiling', 'central', 'could', 'corner', 'da', 'dated', 'del', 'dell', 'della', 'der', 'du', 'evident', 'f', 'example', 'famous', 'five', 'foreground', 'many',
    'fully', 'g', 'good', 'great', 'high', 'however', 'ii', 'important', 'influence', 'le', 'like', 'little', 'lot', 'lotto', 'often', 'painters', 'placed', 'probably', 'use',
    'produced', 'represented', 'represents', 'representing', 'shown', 'six', 'time', 'toward', 'towards', 'twenty', 'way', 'works', 'depicted', 'depicts', 'different', 'see',
    'us', 'career', 'collection', 'composition', 'construction', 'early', 'even', 'exchange', 'master', 'middle', 'museum', 'rard', 'rati', 'several', 'sts', 'ry', 'version',
    'viewer', 'year', 'years', '25', '35', '6', 'among', 'along', 'appear', 'attributed', 'back', 'based', 'became', 'began', 'belonged', 'beside', 'bottom', 'called', 'calling', 'came',
    'characters', 'cm', 'compartments', 'created', 'dei', 'depicting', 'detail', 'displayed', 'either', 'el', 'earlier', 'es', 'especially', 'existence', 'features', 'exterior', 'interior',
    'holding', 'identified', 'intended', 'length', 'lent', 'lived', 'long', 'ly', 'madame', 'much', 'near', 'perhaps', 'place', 'point', 'popular', 'portrayed', 'r', 'rather', 'rd',
    'raire', 'register', 'return', 'reverse', 'rooms', 'seem', 'show', 'sides', 'similar','something', 'spectator', 'state', 'study', 'subjects', 'surrounded', 'sz',
    'taken', 'though', 'together', 'upper', 'v', 'von', 'without', 'would', 'x'
]

def filter_proc(_words, most_common=15):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in _words if word not in stop_words and word not in string.punctuation and word not in bad_words]

    word_counts = Counter(filtered_words)

    most_common_words = word_counts.most_common(most_common)
    most_common_words = [n for (n, c) in most_common_words]
    return most_common_words
    
from tqdm import tqdm
# for each query, there is a different number of relevant museums.
# to compute it, we first compute this huge matrix and then sum over the rows
# NB: it is defined on the number of concepts, since gt_concepts already incorporates that
def groundtruth_relevance_matrix_per_query(gt_concepts):
    rel_mat = np.zeros((len(gt_concepts), len(gt_concepts)))
    for i, cc in enumerate(gt_concepts):
        for j, cc2 in enumerate(gt_concepts):
            rel = get_relevance_values(cc, [cc2])
            assert len(rel) == 1
            rel_mat[i][j] = rel[0]
    
    return rel_mat

   
    
    
    
def main_proc(_loader, _model_desc_pov, _model_pov, _phase, _indices, _eval, _scheduler, _loss_fn, _loss_fn_room, _optimizer, _room_loss_weight):
    total_loss = 0
    num_batches = 0
    
    output_description_total = torch.empty(len(_indices[_phase]), output_feature_size)
    output_pov_total = torch.empty(len(_indices[_phase]), output_feature_size)
    
    with torch.set_grad_enabled(_phase == 'train'):
        for i, batch_data in enumerate(_loader):
            #if _phase == "train":
            #    if len(batch_data) == 9:
            #        data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov, inner_lengths, outer_lengths = batch_data
            #    elif len(batch_data) == 7:
            #        data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov = batch_data
            #    else:
            #        data_desc_pov, data_pov, data_art, names, indexes, len_pov = batch_data
            #else:
            #if len(batch_data) == 8:
            #    data_desc_pov, data_pov, data_art, names, indexes, len_pov, inner_lengths, outer_lengths = batch_data
            #elif len(batch_data) == 7:
            #    data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov = batch_data
            #else:
            #    data_desc_pov, data_pov, data_art, names, indexes, len_pov = batch_data
            if len(batch_data) == 9:
                data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov, inner_lengths, outer_lengths = batch_data
            elif len(batch_data) == 7:
                data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov = batch_data
            else:
                print(len(batch_data))
                assert False
            data_desc_pov = data_desc_pov.to(device)
            data_pov = data_pov.to(device)
            data_art = data_art.to(device)
            # print(inner_lengths, outer_lengths)

            # @DONE
            # The dataloader, using collate_fn, will provide two additional variables, inner_lengths and outer_lengths.
            # inner_lengths -> number of sentences paired to each "room" (*) in each museum.
            # outer_lengths -> number of rooms per museum (+ empty/initial room/sentence).
            # (*starting from the first which contains only one "sentence": In the museum there are ...)
            # NB data_desc_pov: (B*, N, F) where B* > B and contains room-level paragraphs; N is max_num_sents overall; F CLIP size.

            #if not _eval:
            #    _optimizer.zero_grad()

            bsz, fts, no_room_times_no_imgs = data_pov.shape

            if isinstance(_model_desc_pov, HierGRUNet):
                output_room_lev_desc, output_desc_pov = _model_desc_pov(data_desc_pov, inner_lengths, outer_lengths)
                
            else:
                output_room_lev_desc = None
                output_desc_pov = _model_desc_pov(data_desc_pov)
                
            # print(output_room_lev_desc.shape)  # (B*, F)
            # print(output_desc_pov.shape)  # (B, F)
            # @DONE model_desc_pov also receives desc_room_indices and inner_lengths, outer_lengths
            #       to be able to separate room-level data; -> room-level and museum-level GRUs
            # @DONE model_desc_pov also returns room_level text vectors
            if not args.no_artexp:
                x_input = torch.cat((
                    data_pov.transpose(1, 2), 
                    data_art.transpose(1, 2)
                ), -1)
            else:
                x_input = data_pov.transpose(1, 2)

            if isinstance(_model_pov, MyHierBaseline_v3):
                if args.other_method is None:
                    output_pov_img_level, output_pov_room_level, output_pov = _model_pov(data_pov, data_art, len_pov, imgs_per_room=args.n_imgs)
                    room_len_pov = len_pov // args.n_imgs
                    tmp_output_pov_room_level = torch.cat([output_pov_room_level[i, :ix] for i, ix in enumerate(room_len_pov)], 0)
                else:
                    tmp_output_pov_room_level = None
                    output_pov = _model_pov(data_pov, data_art, len_pov, imgs_per_room=args.n_imgs)
                # print(tmp_output_pov_room_level.shape)  # (B*, F)
            
            elif isinstance(_model_pov, MVCNN):
                tmp_output_pov_room_level = None
                _, _, output_pov = _model_pov(data_pov, data_art, list_length=len_pov, clip_mask=None, imgs_per_room=args.n_imgs)
            
            elif isinstance(_model_pov, MVCNN_MVP):
                tmp_output_pov_room_level = None
                output_pov, _ = _model_pov(batch_size=data_pov.shape[0], max_num_views=data_pov.shape[2], num_views=len_pov, x=x_input.float())
            
            elif isinstance(_model_pov, MVCNNSA):
                tmp_output_pov_room_level = None
                _, _, output_pov = _model_pov(data_pov, data_art, list_length=len_pov, clip_mask=None, imgs_per_room=args.n_imgs)

            elif isinstance(_model_pov, DAN):
                tmp_output_pov_room_level = None
                output_pov, _ = _model_pov(batch_size=data_pov.shape[0], max_num_views=data_pov.shape[2], num_views=len_pov, x=x_input.float())
                
            elif isinstance(_model_pov, VSFormer):
                tmp_output_pov_room_level = None
                output_pov, _ = _model_pov(batch_size=data_pov.shape[0], max_num_views=data_pov.shape[2], num_views=len_pov, x=x_input.float())

            else:
                tmp_output_pov_room_level = None
                output_pov = _model_pov(data_pov, x_art=data_art if not args.no_artexp else None)

            """ print(outer_lengths, sum(outer_lengths))
            print(len_pov // 96)
            bbb = [(o -l).item() for o, l in zip(outer_lengths, len_pov // 96)]
            print([i for i, cn in enumerate(bbb) if cn == 0])
            for ii in [i for i, cn in enumerate(bbb) if cn == 0]:
                print("=====")
                print(raw_descs[ii], names[ii], outer_lengths[ii], len_pov[ii] // 96)
            print(output_room_lev_desc.shape, tmp_output_pov_room_level.shape, sum(len_pov // 96)) """
            if len(output_desc_pov.shape) != 2:
                output_desc_pov = output_desc_pov.unsqueeze(0)
            multiplication_dp = cosine_sim(output_desc_pov, output_pov)
            if output_room_lev_desc is not None and tmp_output_pov_room_level is not None:
                multiplication_dp_room = cosine_sim(output_room_lev_desc, tmp_output_pov_room_level)

            if _eval:
                initial_index = i * batch_size
                final_index = (i + 1) * batch_size
                if final_index > len(_indices[_phase]):
                    final_index = len(_indices[_phase])

                output_description_total[initial_index:final_index, :] = output_desc_pov
                output_pov_total[initial_index:final_index, :] = output_pov

        return output_description_total, output_pov_total

    

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence

def evaluate_ndcg(_model_pov, _model_desc_pov, _r1s, _r5s, _r10s, _medrs, _ndcgs, _test_dataset, output_description_test, output_pov_test, skip_ndcg=False, ndcg_10concepts_only=False, gt_rel_mat=None):

#     batch_size = 64
#     test_loader = DataLoader(_test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)
#     test_names = list()
#     _model_pov.eval()
#     _model_desc_pov.eval()
#     output_description_test = torch.empty(len(indices['test']), 256)
#     output_pov_test = torch.empty(len(indices['test']), 256)
#     with torch.no_grad():
#         for j, batch_data in enumerate(test_loader):
            
#             if len(batch_data) == 9:
#                 data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov, inner_lengths, outer_lengths = batch_data
#             elif len(batch_data) == 8:
#                 data_desc_pov, data_pov, data_art, names, indexes, len_pov, inner_lengths, outer_lengths = batch_data
#             elif len(batch_data) == 7:
#                 data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov = batch_data
#             else:
#                 data_desc_pov, data_pov, data_art, names, indexes, len_pov = batch_data
                
#             data_desc_pov = data_desc_pov.to(device)
#             data_pov = data_pov.to(device)
#             data_art = data_art.to(device)

#             test_names.extend(names)

#             bsz, fts, no_room_times_no_imgs = data_pov.shape

#             if not isinstance(_model_desc_pov, GRUNet):
#                 output_room_lev_desc, output_desc_pov = _model_desc_pov(data_desc_pov, inner_lengths, outer_lengths)
#             else:
#                 output_desc_pov = _model_desc_pov(data_desc_pov)
            
#             output_pov_img_level, output_pov_room_level, output_pov = _model_pov(data_pov, data_art, len_pov, imgs_per_room=12)

#             initial_index = j * batch_size
#             final_index = (j + 1) * batch_size
#             if final_index > len(indices['test']):
#                 final_index = len(indices['test'])
#             output_description_test[initial_index:final_index, :] = output_desc_pov
#             output_pov_test[initial_index:final_index, :] = output_pov

    r1, r5, r10, _, _, _, _, _, medr, _ = evaluate(output_description_test, output_pov_test, "test")

    
    output_pov_test_gpu = output_pov_test.to(device)
    output_description_test_gpu = output_description_test.to(device)
    if not skip_ndcg:
        which_concepts_nums = [3, 5, 10, 15] if not ndcg_10concepts_only else [10]
        for it, n_most_common in enumerate(which_concepts_nums):
            gt_concepts = []
            for ix in range(50):
                raw_desc = _test_dataset[ix][-3]
                words = word_tokenize(raw_desc.lower())

                most_common_words = filter_proc(words, n_most_common)
                gt_concepts.append(", ".join(most_common_words))

            # compute nDCG@5,10,15 using n_most_common concepts
            rel_nums = (gt_rel_mat[f"{n_most_common}"] > 0.5).sum(1)
            next_thr = 0.5
            
            
            #for n_top_k in [5, 10, 15, 25, 50]: #, 50]:
            for n_top_k in [5, 10, 15]: #, 50]:
                concepts_as_in_ranking_list = []
                for ix in range(50):  #tqdm(range(50), total=50):
                    query_fts = output_description_test_gpu[ix:ix+1]
            
                    ext_res = torch.topk(cosine_sim(output_pov_test_gpu, query_fts.float()), k=min(n_top_k, rel_nums[ix].item()), dim=0)

                    this_query_concepts  = []
                    for other_ix in ext_res[1]:
                        rd = _test_dataset[other_ix.item()][-3]
                        words = word_tokenize(rd.lower())
                        most_common_words = filter_proc(words, n_most_common)
                        this_query_concepts.append(", ".join(most_common_words))
                    concepts_as_in_ranking_list.append(this_query_concepts)
                    
                    
#             _tmp_dbg = []
#             for n_top_k in [5, 10, 15, 25, 50]: #, 50]:

#                 concepts_as_in_ranking_list = []
#                 for ix in tqdm(range(50), total=50):
#                     query = _test_dataset[ix][0]
#                     # print(query.shape)   # torch.Size([117, 512])
                    
#                     desc_room_indices = _test_dataset[ix][3]
#                     # print("desc room indices", desc_room_indices)  # [1, 24, 40, 58, 82]
                    
#                     inner_lengths = [ix_e - ix_s for ix_s, ix_e in zip([0] + desc_room_indices, desc_room_indices + [len(query)])]
#                     # print("inner_lengths", inner_lengths)  # [1, 23, 16, 18, 24, 35]
                    
#                     outer_lengths = [len(inner_lengths)]
#                     # print("outer_lengths", outer_lengths)  # 6
                    
#                     query_split = [query[ix_s:ix_e] for ix_s, ix_e in zip([0] + inner_lengths, inner_lengths + [-1])]
#                     query_split = pad_sequence(query_split, batch_first=True)
#                     descs_pov = pack_padded_sequence(query_split,
#                                                      torch.tensor(inner_lengths),
#                                                      batch_first=True,
#                                                      enforce_sorted=False)
                    
#                     # print("descs_pov", pad_packed_sequence(descs_pov)[0].shape)  # torch.Size([35, 6])
                    
#                     _room_lev_, query_fts = _model_desc_pov(descs_pov.to(device), inner_lengths, outer_lengths)
#                     # print(output_pov_test.dtype, output_pov_test.device, query_fts.dtype, query_fts.device)
                    
#                     _tmp_dbg.append(query_fts)

#                     ext_res = torch.topk(cosine_sim(output_pov_test.to(device), query_fts.float()), k=min(n_top_k, rel_nums[ix].item()), dim=0)

#                     this_query_concepts  = []
#                     for other_ix in ext_res[1]:
#                         rd = _test_dataset[other_ix.item()][-3]
#                         words = word_tokenize(rd.lower())
#                         most_common_words = filter_proc(words, n_most_common)
#                         this_query_concepts.append(", ".join(most_common_words))
#                     concepts_as_in_ranking_list.append(this_query_concepts)
                    
#                     if ix == 4:
#                         print([_t.shape for _t in _tmp_dbg])
#                         print([_t.sum(1) for _t in _tmp_dbg])
#                         print([_t.shape for _t in output_description_test[:5]])
#                         print([_t.sum() for _t in output_description_test[:5]])
#                         assert False

                ndcg_values = []
                for gt_c, pred_c in zip(gt_concepts, concepts_as_in_ranking_list):
                    rel_scores = get_relevance_values(gt_c, pred_c)
                    gt_rel_scores = get_relevance_values(gt_c, gt_concepts)
                    # print(f"{ndcg(rel_scores)}")
                    ndcg_values.append(ndcg(rel_scores, gt_rel_scores))
                ndcg_values = np.array(ndcg_values)
                # which_ndcgs[it].append(ndcg_values.mean()*100)
                if n_top_k == 50:
                    _ndcgs[f"{n_most_common}"][f"ALL{next_thr}"].append(ndcg_values.mean()*100)
                else:
                    _ndcgs[f"{n_most_common}"][f"{n_top_k}"].append(ndcg_values.mean()*100)


            ##### MUSEUM 2 TEXT
            for n_top_k in [5, 10, 15]: #, 50]:
                concepts_as_in_ranking_list = []
                for ix in range(50):  #tqdm(range(50), total=50):
                    mus_fts = output_pov_test_gpu[ix:ix+1]
            
                    ext_res = torch.topk(cosine_sim(output_description_test_gpu.float(), mus_fts), k=min(n_top_k, rel_nums[ix].item()), dim=0)

                    this_query_concepts  = []
                    for other_ix in ext_res[1]:
                        rd = _test_dataset[other_ix.item()][-3]
                        words = word_tokenize(rd.lower())
                        most_common_words = filter_proc(words, n_most_common)
                        this_query_concepts.append(", ".join(most_common_words))
                    concepts_as_in_ranking_list.append(this_query_concepts)
                    
                ndcg_values = []
                for gt_c, pred_c in zip(gt_concepts, concepts_as_in_ranking_list):
                    rel_scores = get_relevance_values(gt_c, pred_c)
                    gt_rel_scores = get_relevance_values(gt_c, gt_concepts)
                    # print(f"{ndcg(rel_scores)}")
                    ndcg_values.append(ndcg(rel_scores, gt_rel_scores))
                ndcg_values = np.array(ndcg_values)
                # which_ndcgs[it].append(ndcg_values.mean()*100)
                if n_top_k == 50:
                    _ndcgs[f"{n_most_common}"][f"m2t_ALL{next_thr}"].append(ndcg_values.mean()*100)
                else:
                    _ndcgs[f"{n_most_common}"][f"m2t_{n_top_k}"].append(ndcg_values.mean()*100)


    r1 = np.array(r1)
    r5 = np.array(r5)
    r10 = np.array(r10)
    medr = np.array(medr)
    
    _r1s.append(r1)
    _r5s.append(r5)
    _r10s.append(r10)
    _medrs.append(medr)
    return _r1s, _r5s, _r10s, _medrs, _ndcgs
    
        
    
####### TEST WITH FULL HIERARTEX
if __name__ == "__main__":

    
    import argparse

    parser = argparse.ArgumentParser(description="Room aggregation configuration")

    parser.add_argument(
        '--room_txt_agg',
        type=str,
        choices=['rnn', 'monornn', 'avg'],
        default='avg',
        help='Textual aggregation strategy for room content (rnn or avg)'
    )

    parser.add_argument(
        '--room_vis_agg',
        type=str,
        choices=['rnn', 'monornn', 'avg'],
        default='avg',
        help='Visual aggregation strategy for room content (rnn or avg)'
    )

    parser.add_argument(
        '--generalist',
        type=str,
        choices=list(GENERALIST_FEAT_SIZE_DICT.keys()),
        default='clip',
        help='Visual aggregation strategy for room content (rnn or avg)'
    )

    parser.add_argument(
        '--artexp',
        type=str,
        choices=list(visual_bb_ftsize_k.keys()),
        default='rn50',
        help='Visual aggregation strategy for room content (rnn or avg)'
    )

    ######## for ablations
    parser.add_argument(
        '--no-hiervis',
        action="store_true",
        default=False,
        help='Visual aggregation strategy for room content (rnn or avg)'
    )

    parser.add_argument(
        '--no-hiertxt',
        action="store_true",
        default=False,
        help='Visual aggregation strategy for room content (rnn or avg)'
    )

    parser.add_argument(
        '--no-artexp',
        action="store_true",
        default=False,
        help='Visual aggregation strategy for room content (rnn or avg)'
    )

    ####### for second dataset
    parser.add_argument(
        '--gaming',
        action="store_true",
        default=False,
        help='Use the gaming dataset instead of the museum one'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate for the optimizer'
    )

    parser.add_argument(
        '--bs',
        type=int,
        default=32,
        help='Batch size for training'
    )

    parser.add_argument(
        '--other-method',
        type=str,
        choices=['DAN', 'VSFormer', 'MVCNN', 'MVCNNSA', 'merge-rooms-museum-repr'],
        default=None,
        help='If specified, use an alternative method instead of HierArtEx.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help=''
    )

    parser.add_argument(
        '--n-tries',
        type=int,
        default=3,
        help=''
    )

    parser.add_argument(
        '--room-loss-weight',
        type=float,
        default=1.,
        help=''
    )

    parser.add_argument(
        '--n-imgs',
        type=int,
        default=32,
        help=''
    )

    parser.add_argument(
        '--pca-de',
        type=int,
        default=0,
        help=''
    )
    
    parser.add_argument(
        '--skip-ndcg',
        action="store_true",
        default=False,
        help=''
    )

    args = parser.parse_args()
    
    
    import wandb
    use_wandb = False
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
    from torch.nn.utils.rnn import pad_packed_sequence
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np
    import time

    from dan import DAN
    from other_methods import MVCNN, MVCNNSA
    from vsformer import VSFormer
    from mvcnn import MVCNN as MVCNN_MVP

    import random
    import string
    run_folder = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    batch_size = args.bs

    visual_backbone = args.artexp
    visual_bb_ftsize = visual_bb_ftsize_k[visual_backbone]
    if args.pca_de > 0:
        visual_bb_ftsize = args.pca_de
    GENERALIST = args.generalist
    GENERALIST_FEAT_SIZE = GENERALIST_FEAT_SIZE_DICT[GENERALIST]
    if GENERALIST == "clip":
        base_path = "./dataset_gaming_VR/filtered_video_game_features/ViT-B-32_laion2b_s34b_b79k"
    else:
        base_path = f"./dataset_gaming_VR/filtered_video_game_features/{GENERALIST}"
    
    indices_gaming_train = np.random.choice(239, 145, replace=False)
    indices_gaming_val = np.random.choice(list(set(range(239)) - set(indices_gaming_train)), 44, replace=False)
    indices_gaming_test = np.array(list(set(range(239)) - set(indices_gaming_train) - set(indices_gaming_val)))
    indices_game = {
        "train": indices_gaming_train, 
        "val": indices_gaming_val, 
        "test": indices_gaming_test
    }

    # TODO change change paths
    #train_dataset = DescriptionSceneMuseum(f"./{base_path}/descriptions/sentences", 
    #                                       f"./{base_path}/descriptions/tokens_strings", 
    #                                       f"./{base_path}/images",
    #                                       f"./gaming_DE_features{'_PCA' + str(args.pca_de) if args.pca_de > 0 else ''}",
    #                                indices_game, "train", num_imgs=args.n_imgs)

    #val_dataset = DescriptionSceneMuseum(f"./{base_path}/descriptions/sentences", 
    #                                       f"./{base_path}/descriptions/tokens_strings", 
    #                                       f"./{base_path}/images",
    #                                       f"./gaming_DE_features{'_PCA' + str(args.pca_de) if args.pca_de > 0 else ''}",
    #                                indices_game, "val", num_imgs=args.n_imgs)

    test_dataset = DescriptionSceneMuseum(f"./{base_path}/descriptions/sentences", 
                                           f"./{base_path}/descriptions/tokens_strings", 
                                           f"./{base_path}/unpacked_images",
                                           f"./gaming_DE_features{'_PCA' + str(args.pca_de) if args.pca_de > 0 else ''}",
                                    indices_game, "test", num_imgs=args.n_imgs)
    
    collate_fn_to_use = collate_fn if not args.no_hiertxt else collate_fn_desc
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_to_use, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_to_use, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_to_use, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    num_epochs = args.epochs
    number_of_tries = args.n_tries
    final_output_strings = []

    output_feature_size = 256 # default: 256
    print(args.room_txt_agg,  'mono' not in args.room_txt_agg)
    is_bidirectional = 'mono' not in args.room_txt_agg
    is_bidirectional_RNN_scenes = 'mono' not in args.room_vis_agg
    room_loss_weight = args.room_loss_weight
    room_txt_agg = args.room_txt_agg  # rnn, avg
    room_vis_agg = args.room_vis_agg  # rnn, avg

    if args.other_method is None:
        if args.no_hiervis or args.no_hiertxt or args.no_artexp:
            approach_name = "hierarchical_v2"
            if not args.no_artexp:
                approach_name += f"_art_vectors_{visual_backbone}"
            if not args.no_hiertxt:
                approach_name += f"_hier_loss_room{room_loss_weight}_txt{room_txt_agg}"
            if not args.no_hiervis:
                approach_name += f"_vis{'bi' if is_bidirectional_RNN_scenes else ''}{room_vis_agg}"
            approach_name += f"_gen_{GENERALIST}"
            
        else:
            approach_name = f"hierarchical_v3_art_vectors_{visual_backbone}_hier_loss_room{room_loss_weight}_vis{'bi' if is_bidirectional_RNN_scenes else ''}{room_vis_agg}_txt{room_txt_agg}_gen_{GENERALIST}"
    
    else:
        approach_name = args.other_method + f"_gen_{GENERALIST}"
        if not args.no_artexp:
            approach_name += f"_art_vectors_{visual_backbone}"
    if not args.no_artexp and args.pca_de > 0:
        approach_name = approach_name.replace(f"_art_vectors_{visual_backbone}", f"_art_vectors_{visual_backbone}PCA{args.pca_de}")
    approach_name = "gaming_" + approach_name

    r1s, r5s, r10s, medrs = [], [], [], []
    #ndcgs = {nc: {k: [] for k in ['5', '10', '15', '25', '50', 'ALL0.5', 'ALL0.25']} for nc in ['5', '10', '15']}
    ndcgs = {nc: {k: [] for k in [
        '5', '10', '15', '25', '50', 'ALL0.5', 'ALL0.25',
        'm2t_5', 'm2t_10', 'm2t_15', 'm2t_25', 'm2t_50', 'm2t_ALL0.5', 'm2t_ALL0.25',
        ]} for nc in ['3', '5', '10', '15']}
    
    # here we pre-compute it for n_concepts=5, 10, 15
    gt_rel_mat = {}
    gt_rel_num = {}
    for n_most_common in [3, 5, 10, 15]:
        gt_concepts = []
        for ix in range(50):  #tqdm(range(50), total=50):
            raw_desc = test_dataset[ix][-3]
            words = word_tokenize(raw_desc.lower())

            most_common_words = filter_proc(words, n_most_common)
            # for _ in range(n_top_k):
            gt_concepts.append(", ".join(most_common_words))
        gt_rel_mat[f"{n_most_common}"] = groundtruth_relevance_matrix_per_query(gt_concepts)
        gt_rel_num[f"{n_most_common}"] = (gt_rel_mat[f"{n_most_common}"] > 0.5).sum(1)


    for n_try in range(number_of_tries):
        lr = args.lr # default: 0.008

        #loss_fn = LossContrastive(approach_name, patience=25, delta=0.0001)
        #loss_fn_room = LossContrastive(approach_name, patience=25, delta=0.0001)

        if args.no_hiertxt:
            model_desc_pov = GRUNet(hidden_size=output_feature_size, num_features=GENERALIST_FEAT_SIZE, is_bidirectional=is_bidirectional)
        else:
            model_desc_pov = HierGRUNet(hidden_size=output_feature_size, num_features=GENERALIST_FEAT_SIZE, is_bidirectional=is_bidirectional, room_txt_agg=room_txt_agg)
        
        if args.other_method is None:        
            if args.no_hiervis:
                __model_arch = "MyBaseline"
                model_pov = MyBaseline(in_channels=GENERALIST_FEAT_SIZE, out_channels=256, feature_size=output_feature_size, art_features_size=visual_bb_ftsize if not args.no_artexp else 0)
            else:
                __model_arch = "MyHierBaseline_v3"
                model_pov = MyHierBaseline_v3(in_channels=GENERALIST_FEAT_SIZE, out_channels=256, feature_size=output_feature_size, art_features_size=visual_bb_ftsize if not args.no_artexp else 0, bidirectional=is_bidirectional_RNN_scenes, room_vis_agg=room_vis_agg)

        else:
            if args.other_method == "DAN":
                __model_arch = "DAN"
                model_pov = DAN(h=2, feature_dim=GENERALIST_FEAT_SIZE, num_heads=4, inner_dim=output_feature_size, dropout=0.1, output_feature_size=output_feature_size, art_features_size=visual_bb_ftsize if not args.no_artexp else 0)
            elif args.other_method == "VSFormer":
                __model_arch = "VSFormer"
                model_pov = VSFormer(feature_dim=GENERALIST_FEAT_SIZE, num_layers=4, num_heads=8, attention_dropout=0.1, mlp_dropout=0.1, widening_factor=2, output_feature_size=output_feature_size, art_features_size=visual_bb_ftsize if not args.no_artexp else 0)
            elif args.other_method == "MVCNN":
                __model_arch = "MVCNN"
                #model_pov = MVCNN(in_channels=GENERALIST_FEAT_SIZE, out_channels=output_feature_size, feature_size=128, art_features_size=visual_bb_ftsize if not args.no_artexp else 0)
                model_pov = MVCNN_MVP(feature_dim=GENERALIST_FEAT_SIZE, output_feature_size=output_feature_size, art_features_size=visual_bb_ftsize if not args.no_artexp else 0)
            elif args.other_method == "MVCNNSA":
                __model_arch = "MVCNNSA"
                model_pov = MVCNNSA(in_channels=GENERALIST_FEAT_SIZE, out_channels=output_feature_size, feature_size=128, art_features_size=visual_bb_ftsize if not args.no_artexp else 0)
            elif args.other_method == "merge-rooms-museum-repr":
                __model_arch = "merge-rooms-museum-repr"
                model_pov = MyHierBaseline_v3(in_channels=GENERALIST_FEAT_SIZE, out_channels=256, feature_size=output_feature_size, art_features_size=visual_bb_ftsize if not args.no_artexp else 0, bidirectional=is_bidirectional_RNN_scenes, room_vis_agg=room_vis_agg, merge_room_museum_repr=True)
            else:
                assert False, args.other_method
        
        #print(model_pov)
        #print(model_desc_pov)
        model_desc_pov.to(device)
        model_pov.to(device)

        """params = list(model_desc_pov.parameters()) + list(model_pov.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        step_size = 27
        gamma = 0.75
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        sched_name = StepLR

        if use_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="Museums-Gaming",

                # track hyperparameters and run metadata
                config={**{
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "architecture": MyHierBaseline_v3 if not args.no_hiervis else MyBaseline,
                    "epochs": num_epochs,
                    "approach_name": approach_name,
                    "output_feature_size": output_feature_size,
                    "scheduler/name": sched_name,
                    "room_loss_weight": room_loss_weight,
                }, **{f"scheduler/{n}": v for n, v in scheduler.__dict__.items()}}
            )

        best_r5 = 0
        print(f"({n_try+1}/{number_of_tries}) Train procedure ...")
        for ep in tqdm(range(num_epochs)):

            model_desc_pov.train()
            model_pov.train()

            _, _ = main_proc(train_loader, model_desc_pov, model_pov, "train", indices_game, False, scheduler, loss_fn, loss_fn_room, optimizer, room_loss_weight)

            model_desc_pov.eval()
            model_pov.eval()

            output_description_val, output_pov_val = main_proc(val_loader, model_desc_pov, model_pov, "val", indices_game, True, scheduler, loss_fn, loss_fn_room, optimizer, room_loss_weight)

            r1, r5, r10, _, _, _, _, _, _, _ = evaluate(output_description=output_description_val,
                                                        output_scene=output_pov_val, section='val',
                                                        out_values=ep % 5 == 4)

            if r5 > best_r5:
                best_r5 = r5
                save_best_model(f"{approach_name}_{n_try}", run_folder, model_pov.state_dict(), model_desc_pov.state_dict())

            if use_wandb:
                wandb.log({
                    "val/T2S_R@1": r1, 
                    "val/T2S_R@5": r5, 
                    "val/T2S_R@10": r10, 
                })

            output_description_val_train, output_pov_val_train = main_proc(train_loader, model_desc_pov, model_pov, "train", indices_game, True, scheduler, loss_fn, loss_fn_room, optimizer, room_loss_weight)

            r1, r5, r10, _, _, _, _, _, _, _ = evaluate(output_description=output_description_val_train,
                                                        output_scene=output_pov_val_train, section='TRAIN',
                                                        out_values=ep % 5 == 4)
            if use_wandb:
                wandb.log({
                    "train/T2S_R@1": r1, 
                    "train/T2S_R@5": r5, 
                    "train/T2S_R@10": r10, 
                })"""

        bm_pov, bm_desc_pov = load_best_model(f"{approach_name}_{n_try}", run_folder)
        model_pov.load_state_dict(bm_pov)
        model_desc_pov.load_state_dict(bm_desc_pov)

        model_pov.eval()
        model_desc_pov.eval()
        output_description_test, output_pov_test = main_proc(test_loader, model_desc_pov, model_pov, "test", indices_game, True, None, None, None, None, room_loss_weight)

        ds1, ds5, ds10, sd1, sd5, sd10, _, _, ds_medr, sd_medr, formatted_string = evaluate(
            output_description=output_description_test,
            output_scene=output_pov_test,
            section="test",
            out_values=False,
            excel_format=True)

        r1s, r5s, r10s, medrs, ndcgs = evaluate_ndcg(model_pov, model_desc_pov, r1s, r5s, r10s, medrs, ndcgs, test_dataset, output_description_test, output_pov_test, gt_rel_mat=gt_rel_mat, skip_ndcg=args.skip_ndcg)

        #if use_wandb:
        #    wandb.log({
        #        "test/T2S_R@1": ds1, 
        #        "test/T2S_R@5": ds5, 
        #        "test/T2S_R@10": ds10,
        #        "test/S2T_R@1": sd1, 
        #        "test/S2T_R@5": sd5, 
        #        "test/S2T_R@10": sd10, 
        #    })

        final_output_strings.append(formatted_string)

    #if use_wandb:
    #    wandb.finish()
    cmp_avg = []
    for out_str in final_output_strings:
        print(out_str)
        cmp_avg.append({k: v for k, v in zip(["tv R1", "tv R5", "tv R10", "vt R1", "vt R5", "vt R10", "tv MedR", "vt MedR"], map(float, out_str.split(",")))})
    
    if len(cmp_avg) == 1:
        ss = ";".join([f"{cmp_avg[0][k]:.1f}" for k in cmp_avg[0].keys()])
    else:
        ss = ";".join([f"{(cmp_avg[0][k]+cmp_avg[1][k]+cmp_avg[2][k])/3:.1f}" for k in cmp_avg[0].keys()])
    print(ss)

    """ print("\t R@1  R@5  R@10  MedR c5k5 c5k10 c5k15 c5k50 c10k5 c10k10 c10k15 c10k50 c15k5 c15k10 c15k15 c15k50")
    run_vals = [r1s, r5s, r10s, medrs] #+ [ndcgs[nc][k] for k in ['5', '10', '15', '50'] for nc in ['5', '10', '15']]
    # save_data_(run_vals, room_vis_agg, room_txt_agg, visual_backbone, "mobile_clip_baseline")
    for i in range(args.n_tries):
        ss = ', '.join([f"{r[i]:.1f}" for r in run_vals])
        print(f"run {i}: {ss}") # {r1s[i]:.1f}, {r5s[i]:.1f}, {r10s[i]:.1f}, {medrs[i]:.1f}, {ndcgs[i]:.1f}, {ndcgs_10[i]:.1f}, {ndcgs_15[i]:.1f}")
    run_vals = [r1s, r5s, r10s, medrs] + [ndcgs[nc][k] for k in ['5', '10', '15', '50'] for nc in ['5', '10', '15']]
    ss = ', '.join([f"{np.array(r).mean():.1f}" for r in run_vals])
    print(f"  AVG: {ss}")

    ### MUSEUM 2 TEXT
    run_vals = [ndcgs[nc][f"m2t_{k}"] for k in ['5', '10', '15', '50'] for nc in ['5', '10', '15']]
    ss = ', '.join([f"{np.array(r).mean():.1f}" for r in run_vals])
    print(f"  M2T: {ss}") """

    print("========= RE-ORDERED =========")
    run_vals = [r1s, r5s, r10s, medrs]
    for nc in ['3', '5', '10', '15']:
        run_vals.extend([ndcgs[nc][k] for k in ['5', '10', '15', '50']])
    ss = ', '.join([f"{np.array(r).mean():.1f}" for r in run_vals])
    print(f"  AVG: {ss}")

    ### MUSEUM 2 TEXT
    run_vals = []
    for nc in ['3', '5', '10', '15']:
        run_vals.extend([ndcgs[nc][f"m2t_{k}"] for k in ['5', '10', '15', '50']])
    # run_vals = [ndcgs[nc][f"m2t_{k}"] for k in ['5', '10', '15', '50'] for nc in ['5', '10', '15']]
    ss = ', '.join([f"{np.array(r).mean():.1f}" for r in run_vals])
    print(f"  M2T: {ss}")
    print()
    print("String for table:")
    get_vts = lambda k: f"{(cmp_avg[0][k]+cmp_avg[1][k]+cmp_avg[2][k])/3:.1f}" if len(cmp_avg) > 1 else f"{cmp_avg[0][k]:.1f}"
    get_ndcgs = lambda nc, k: f"{np.array(ndcgs[nc][k]).mean():.1f}"
    print(get_vts("tv R1"), get_vts("tv R5"), get_vts("tv R10"))
    print(" & ".join([
        get_vts("tv R1"), get_vts("tv R5"), get_vts("tv R10"), get_vts("tv MedR"),
        get_ndcgs('3', '10'), get_ndcgs('5', '10'), get_ndcgs('10', '10'), 
        get_vts("vt R1"), get_vts("vt R5"), get_vts("vt R10"), get_vts("vt MedR"),
        get_ndcgs('3', 'm2t_10'), get_ndcgs('5', 'm2t_10'), get_ndcgs('10', 'm2t_10'), 
        ]))

    print()




