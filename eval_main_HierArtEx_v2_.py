import os 
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


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("="*10, "Using", device)

import pandas as pd
indices = pd.read_pickle("indices_museum_dataset.pkl")
indices['train'][:10], indices['val'][:10], indices['test'][:10],

from torch.utils.data import Dataset

class DescriptionSceneMuseum(Dataset):
    def __init__(self, data_description_path, data_raw_description_path, data_scene_path, data_art_path, indices, split, load_art_vectors=False):
        self.description_path = data_description_path
        self.raw_description_path = data_raw_description_path
        self.data_pov_path = data_scene_path
        self.indices = indices[split]
        self.split = split
        self.load_art_vectors = load_art_vectors

        available_data = [im.strip(".pt") for im in os.listdir(data_scene_path)]
        available_data = sorted(available_data)
        available_data = [available_data[ix] for ix in self.indices.tolist()]

        self.descs = [torch.load(os.path.join(data_description_path, f"{sm}.pt"), weights_only=True) for sm in available_data]
        self.raw_descs = [" ".join(pd.read_pickle(os.path.join(data_raw_description_path, f"{sm}.pkl"))) for sm in available_data]
        self.room_desc_indices = [[
                sent_ix for sent_ix, sent in enumerate(sm.split("."))
                if re.match(r"^ In the \w+ room , there are", sent)
        ] for sm in self.raw_descs]
        self.pov_images = [torch.load(os.path.join(data_scene_path, f"{sm}.pt"), weights_only=True) for sm in available_data]
        if self.load_art_vectors:
            self.art_vectors = [torch.load(os.path.join(data_art_path, f"{sm}.pt"), weights_only=True) for sm in available_data]
        else:
            self.art_vectors = []

        self.names = available_data
        print(f"'{split.upper()}': {len(self.names)} names, "
              f"{len(self.descs)} sentences ({sum([len(x) for x in self.descs]) / len(self.descs)} avg), "
              f"{len(self.pov_images)} images ({sum([len(x) for x in self.pov_images]) / len(self.pov_images)} avg).")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        desc_tensor = self.descs[index]
        # if self.split == "train":
        raw_desc = self.raw_descs[index]
        scene_img_tensor = self.pov_images[index]
        if self.load_art_vectors:
            scene_art_tensor = self.art_vectors[index]
        else:
            scene_art_tensor = torch.zeros_like(scene_img_tensor)
        name = self.names[index]
        room_desc_indices = self.room_desc_indices[index]


        return desc_tensor, scene_img_tensor, scene_art_tensor, room_desc_indices, raw_desc, name, index
        
visual_bb_ftsize_k = {'rn18': 512, 'rn34': 512, 'rn50': 2048, 'rn101': 2048, 'vitb16': 768, 'vitb32': 768, 'openclip': 512}

GENERALIST_FEAT_SIZE_DICT = {
    'clip': 512,
    'siglip': 768,
    # 'blip': 512,
    'blip_base': 768,
    'mobile_clip': 512
}



import re


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
        # print("-"*5)
        # print("{ds_r1};{ds_r5};{ds_r10};{sd_r1};{sd_r5};{sd_r10};{ds_medr};{sd_medr}")
        # print(f"{ds_r1};{ds_r5};{ds_r10};{sd_r1};{sd_r5};{sd_r10};{ds_medr};{sd_medr}")
        # print("-"*5)
        formatted_string = f"{ds_r1};{ds_r5};{ds_r10};{sd_r1};{sd_r5};{sd_r10};{ds_medr};{sd_medr}"
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

def load_this_model(model_path):
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
        x1_ragged = [h_n_room[ol_s:ol_e] for ol_s, ol_e in aa]  # -> (B, N, F)
        if self.room_txt_agg in ['rnn', 'monornn']:
            x1 = pad_sequence(x1_ragged, batch_first=True)
            x1 = pack_padded_sequence(x1,
                                      torch.tensor(outer_lens),
                                      batch_first=True,
                                      enforce_sorted=False)
            _, h_n_museum = self.gru_museum(x1)
            if self.is_bidirectional:
                return out_h_n_room, h_n_museum.mean(0), x1_ragged[:-1]
            else:
                return out_h_n_room, h_n_museum.squeeze(0), x1_ragged[:-1]
        else:
            # print([a.shape for a in x1])
            x1 = pad_sequence(x1_ragged, batch_first=True)[:-1]
            h_n_museum = x1.sum(-2) / torch.tensor(outer_lens, device=x1.device).unsqueeze(1)
        return out_h_n_room.squeeze(0), h_n_museum.squeeze(0), x1_ragged[:-1]
    


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
        
        list_length_t = torch.tensor(list_length, device=x1.device) if isinstance(list_length, list) else list_length.to(x1.device)
        x1 = x1.sum(1) / list_length_t.unsqueeze(1)
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
nltk.download('stopwords')
nltk.download('punkt')

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

            assert len(batch_data) == 9 or len(batch_data) == 7
            if len(batch_data) == 9:
                data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov, inner_lengths, outer_lengths = batch_data
            else:
                data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov = batch_data

            data_desc_pov = data_desc_pov.to(device)
            data_pov = data_pov.to(device)
            data_art = data_art.to(device)


            bsz, fts, no_room_times_no_imgs = data_pov.shape

            if isinstance(_model_desc_pov, HierGRUNet):
                output_room_lev_desc, output_desc_pov, output_room_lev_desc_ragged = _model_desc_pov(data_desc_pov, inner_lengths, outer_lengths)
                
            else:
                output_room_lev_desc = None
                output_desc_pov = _model_desc_pov(data_desc_pov)
                
            if not args.no_artexp:
                x_input = torch.cat((
                    data_pov.transpose(1, 2), 
                    data_art.transpose(1, 2)
                ), -1)
            else:
                x_input = data_pov.transpose(1, 2)

            if isinstance(_model_pov, MyHierBaseline_v3):
                if args.other_method is None:
                    output_pov_img_level, output_pov_room_level, output_pov = _model_pov(data_pov, data_art, len_pov, imgs_per_room=12)
                    room_len_pov = len_pov // 12
                    tmp_output_pov_room_level = torch.cat([output_pov_room_level[i, :ix] for i, ix in enumerate(room_len_pov)], 0)
                else:
                    tmp_output_pov_room_level = None
                    output_pov = _model_pov(data_pov, data_art, len_pov, imgs_per_room=12)
                # print(tmp_output_pov_room_level.shape)  # (B*, F)
            
            elif isinstance(_model_pov, MVCNN):
                tmp_output_pov_room_level = None
                _, _, output_pov = _model_pov(data_pov, data_art, list_length=len_pov, clip_mask=None, imgs_per_room=12)
            
            elif isinstance(_model_pov, MVCNN_MVP):
                tmp_output_pov_room_level = None
                output_pov, _ = _model_pov(batch_size=data_pov.shape[0], max_num_views=data_pov.shape[2], num_views=len_pov, x=x_input.float())
            
            elif isinstance(_model_pov, MVCNNSA):
                tmp_output_pov_room_level = None
                _, _, output_pov = _model_pov(data_pov, data_art, list_length=len_pov, clip_mask=None, imgs_per_room=12)

            elif isinstance(_model_pov, DAN):
                tmp_output_pov_room_level = None
                output_pov, _ = _model_pov(batch_size=data_pov.shape[0], max_num_views=data_pov.shape[2], num_views=len_pov, x=x_input.float())
                
            elif isinstance(_model_pov, VSFormer):
                tmp_output_pov_room_level = None
                output_pov, _ = _model_pov(batch_size=data_pov.shape[0], max_num_views=data_pov.shape[2], num_views=len_pov, x=x_input.float())

            else:
                tmp_output_pov_room_level = None
                output_pov = _model_pov(data_pov, data_art if not args.no_artexp else None, len_pov)

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


    r1, r5, r10, _, _, _, _, _, medr, _ = evaluate(output_description_test, output_pov_test, "test")

    
    output_pov_test_gpu = output_pov_test.to(device)
    output_description_test_gpu = output_description_test.to(device)
    if not skip_ndcg:
        which_concepts_nums = [3, 5, 10, 15] if not ndcg_10concepts_only else [10]
        for it, n_most_common in enumerate(which_concepts_nums):
            gt_concepts = []
            for ix in range(450):
                raw_desc = _test_dataset[ix][-3]
                words = word_tokenize(raw_desc.lower())

                most_common_words = filter_proc(words, n_most_common)
                gt_concepts.append(", ".join(most_common_words))

            # compute nDCG@5,10,15 using n_most_common concepts
            rel_nums = (gt_rel_mat[f"{n_most_common}"] > 0.5).sum(1)
            next_thr = 0.5
            
            
            for n_top_k in [5, 10, 15]: #, 450]:
                concepts_as_in_ranking_list = []
                for ix in tqdm(range(450), total=450):
                    query_fts = output_description_test_gpu[ix:ix+1]
            
                    ext_res = torch.topk(cosine_sim(output_pov_test_gpu, query_fts.float()), k=min(n_top_k, rel_nums[ix].item()), dim=0)

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
                if n_top_k == 450:
                    _ndcgs[f"{n_most_common}"][f"ALL{next_thr}"].append(ndcg_values.mean()*100)
                else:
                    _ndcgs[f"{n_most_common}"][f"{n_top_k}"].append(ndcg_values.mean()*100)


            ##### MUSEUM 2 TEXT
            for n_top_k in [5, 10, 15]: #, 450]:
                concepts_as_in_ranking_list = []
                for ix in range(450):  #tqdm(range(450), total=450):
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
                if n_top_k == 450:
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

    parser.add_argument(
        '--lambda-rooms',
        type=float,
        default=1.0,
        help='Weight for the room-level loss'
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

    parser.add_argument(
        '--skip-ndcg',
        action="store_true",
        default=False,
        help='Visual aggregation strategy for room content (rnn or avg)'
    )

    parser.add_argument(
        '--room-loss-weight',
        type=float,
        default=1.,
        help='Weight for the room-level loss component'
    )

    parser.add_argument(
        '--other-method',
        type=str,
        choices=['DAN', 'VSFormer', 'MVCNN', 'MVCNNSA', 'merge-rooms-museum-repr'],
        default=None,
        help='If specified, use an alternative method instead of HierArtEx.'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
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
        '--lr',
        type=float,
        default=0.001,
        help=''
    )

    parser.add_argument(
        '--resume-ckpts',
        nargs="+",
        default=[],
        help=''
    )

    parser.add_argument(
        '--eval-user-queries',
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
    run_folder = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    batch_size = args.batch_size

    visual_backbone = args.artexp
    visual_bb_ftsize = visual_bb_ftsize_k[visual_backbone]
    GENERALIST = args.generalist
    GENERALIST_FEAT_SIZE = GENERALIST_FEAT_SIZE_DICT[GENERALIST]
    if GENERALIST == "clip":
        base_path = "./tmp_museums/open_clip_features_museums3k"
    else:
        base_path = f"./tmp_museums/museums3k_new_features/{GENERALIST}"

    test_dataset = DescriptionSceneMuseum(f"./{base_path}/descriptions/sentences", 
                                           f"./{base_path}/descriptions/tokens_strings", 
                                           f"./{base_path}/images",
                                           f"./preextracted_vectors_wikiart_{visual_backbone}",
                                    indices, "test", load_art_vectors=not args.no_artexp)
    
    collate_fn_to_use = collate_fn if not args.no_hiertxt else collate_fn_desc
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_to_use, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    num_epochs = args.epochs
    number_of_tries = args.n_tries
    final_output_strings = []

    output_feature_size = 256 # default: 256
    is_bidirectional = 'mono' not in args.room_txt_agg
    is_bidirectional_RNN_scenes = 'mono' not in args.room_vis_agg
    room_loss_weight = args.lambda_rooms
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
    print("=== SHOULD BE:", approach_name)
        

    if args.eval_user_queries:

        def create_rank_query(result, entire_descriptor, desired_output_index):
            similarity_desc_query = torch.nn.functional.cosine_similarity(entire_descriptor, result, dim=1)
            similarity_desc_query = similarity_desc_query.squeeze()

            sorted_indices = torch.argsort(similarity_desc_query, descending=True)
            if desired_output_index.dim() == 0:
                positions = (sorted_indices.unsqueeze(0) == desired_output_index.unsqueeze(0))
            else:
                positions = (sorted_indices.unsqueeze(0) == desired_output_index.unsqueeze(1))
            match_positions = torch.nonzero(positions, as_tuple=False)
            lowest_rank = match_positions[:, 1].min()

            return lowest_rank.item(), sorted_indices


        def evaluate_query_train(output_query, output, section, relevance_matrix=None, subset_queries=-1, no_queries_per_image=1):
            avg_rank_query = 0
            ranks_query = []
            if subset_queries > 0:
                output_query = output_query[:subset_queries]

            for j, i in enumerate(output_query):
                rank, sorted_list = create_rank_query(i, output, torch.nonzero(relevance_matrix[j // no_queries_per_image, :]).squeeze())

                avg_rank_query += rank
                ranks_query.append(rank)

            ranks_query = np.array(ranks_query)

            n_q = len(output_query)
            r1 = 100 * len(np.where(ranks_query < 1)[0]) / n_q
            r5 = 100 * len(np.where(ranks_query < 5)[0]) / n_q
            r10 = 100 * len(np.where(ranks_query < 10)[0]) / n_q
            medr = np.median(ranks_query) + 1
            meanr = ranks_query.mean() + 1

            qd_out = ""
            for mn, mv in [("R@1", r1),
                        ("R@5", r5),
                        ("R@10", r10),
                        ("median rank", medr),
                        ("mean rank", meanr),
                        ]:
                qd_out += f"{mn}: {mv:.4f}   "

            """print(section + " data: ")
            print("Queries ranking: " + qd_out)

            print("-" * 5)
            print("{qd_r1};{qd_r5};{qd_r10};{qd}")"""
            print(f"{r1};{r5};{r10};{medr}")
            """print("-" * 5)"""
            formatted_string = f"{r1};{r5};{r10};{medr}"
            return r1, r5, r10, medr, formatted_string

        
    r1s, r5s, r10s, medrs = [], [], [], []
    ndcgs = {nc: {k: [] for k in [
        '5', '10', '15', '25', '50', 'ALL0.5', 'ALL0.25',
        'm2t_5', 'm2t_10', 'm2t_15', 'm2t_25', 'm2t_50', 'm2t_ALL0.5', 'm2t_ALL0.25',
        ]} for nc in ['3', '5', '10', '15']}
    
    gt_rel_mat = {}
    gt_rel_num = {}
    for n_most_common in [3, 5, 10, 15]:
        gt_concepts = []
        for ix in tqdm(range(450), total=450):
            raw_desc = test_dataset[ix][-3]
            words = word_tokenize(raw_desc.lower())

            most_common_words = filter_proc(words, n_most_common)
            gt_concepts.append(", ".join(most_common_words))
        gt_rel_mat[f"{n_most_common}"] = groundtruth_relevance_matrix_per_query(gt_concepts)
        gt_rel_num[f"{n_most_common}"] = (gt_rel_mat[f"{n_most_common}"] > 0.5).sum(1)


        
        
    for n_try in range(number_of_tries):

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
        model_desc_pov.to(device)
        model_pov.to(device)


        if len(args.resume_ckpts) > 0:
            bm_pov, bm_desc_pov = load_this_model(args.resume_ckpts[n_try])
            print("Loaded from:", args.resume_ckpts[n_try])

            if 'trf_photo.weight' in bm_pov.keys() and 'trf_mean.weight' in bm_pov.keys():
                print(bm_pov.keys())
                # need to update names due to old codebase
                bm_pov['trf.weight'] = bm_pov['trf_photo.weight']
                bm_pov['trf.bias'] = bm_pov['trf_photo.bias']
                bm_pov['fc.weight'] = bm_pov['trf_mean.weight']
                bm_pov['fc.bias'] = bm_pov['trf_mean.bias']
                bm_pov.pop("trf_photo.weight", None)
                bm_pov.pop("trf_photo.bias", None)
                bm_pov.pop("trf_mean.bias", None)
                bm_pov.pop("trf_mean.weight", None)
        else:
            bm_pov, bm_desc_pov = load_best_model(f"{approach_name}_{n_try}", run_folder)
        model_pov.load_state_dict(bm_pov)
        model_desc_pov.load_state_dict(bm_desc_pov)

        model_pov.eval()
        model_desc_pov.eval()
        output_description_test, output_pov_test = main_proc(test_loader, model_desc_pov, model_pov, "test", indices, True, None, None, None, None, None)

        if args.eval_user_queries:
            print("="*10, f'QUERIES (run {n_try})', "="*10)
            print("results for 50, 100, ..., 550 queries (n=50 used in paper, avg over the three runs)")
            data_queries = torch.load(f"./data_query_test{'_mobile_clip' if args.generalist == 'mobile_clip' else ''}.pt", weights_only=True).to(device)
            relevance = torch.load('./relevance_matrix.pt', weights_only=True).to(device)
            if isinstance(model_desc_pov, HierGRUNet):
                _, output_query_test, _ = model_desc_pov(data_queries, None, [1]*data_queries.shape[0])
                
            else:
                output_query_test = model_desc_pov(data_queries)

            for n_queries in range(50, 600, 50):
                qs1, qs5, qs10, sq1, sq5 = evaluate_query_train(output_query=output_query_test.to(device),
                                                                            output=output_pov_test.to(device),
                                                                            relevance_matrix=relevance.to(device),
                                                                            no_queries_per_image=3,
                                                                            subset_queries=n_queries,
                                                                            section='test')
                
                #print(f"{n_queries} QUERY TO SCENE: R@1: {qs1}, R@5: {qs5}, R@10: {qs10}")

        ds1, ds5, ds10, sd1, sd5, sd10, _, _, ds_medr, sd_medr, formatted_string = evaluate(
            output_description=output_description_test,
            output_scene=output_pov_test,
            section="test",
            out_values=False,
            excel_format=True)

        r1s, r5s, r10s, medrs, ndcgs = evaluate_ndcg(model_pov, model_desc_pov, r1s, r5s, r10s, medrs, ndcgs, test_dataset, output_description_test, output_pov_test, gt_rel_mat=gt_rel_mat, skip_ndcg=args.skip_ndcg)
        

        final_output_strings.append(formatted_string)

    cmp_avg = []
    for out_str in final_output_strings:
        print(out_str)
        cmp_avg.append({k: v for k, v in zip(["tv R1", "tv R5", "tv R10", "vt R1", "vt R5", "vt R10", "tv MedR", "vt MedR"], map(float, out_str.split(";")))})
    
    if len(cmp_avg) == 1:
        ss = ";".join([f"{cmp_avg[0][k]:.1f}" for k in cmp_avg[0].keys()])
    else:
        ss = ";".join([f"{(cmp_avg[0][k]+cmp_avg[1][k]+cmp_avg[2][k])/3:.1f}" for k in cmp_avg[0].keys()])
    print(ss)


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
    print()



