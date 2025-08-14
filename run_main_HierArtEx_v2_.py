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
print("="*10, "Using", device)

import pandas as pd
indices = pd.read_pickle("indices_museum_dataset.pkl")
indices['train'][:10], indices['val'][:10], indices['test'][:10],

from torch.utils.data import Dataset

class DescriptionSceneMuseum(Dataset):
    def __init__(self, data_description_path, data_raw_description_path, data_scene_path, data_art_path, indices, split, customized_margin=False):
        self.description_path = data_description_path
        self.raw_description_path = data_raw_description_path
        self.data_pov_path = data_scene_path
        self.indices = indices[split]
        self.split = split

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
        self.art_vectors = [torch.load(os.path.join(data_art_path, f"{sm}.pt"), weights_only=True) for sm in available_data]
        self.names = available_data
        print(f"'{split.upper()}': {len(self.names)} names, "
              f"{len(self.descs)} sentences ({sum([len(x) for x in self.descs]) / len(self.descs)} avg), "
              f"{len(self.pov_images)} images ({sum([len(x) for x in self.pov_images]) / len(self.pov_images)} avg).")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        desc_tensor = self.descs[index]
        if self.split == "train":
            raw_desc = self.raw_descs[index]
        scene_img_tensor = self.pov_images[index]
        scene_art_tensor = self.art_vectors[index]
        name = self.names[index]
        room_desc_indices = self.room_desc_indices[index]

        if self.split == "train":
            return desc_tensor, scene_img_tensor, scene_art_tensor, room_desc_indices, raw_desc, name, index
        else:
            return desc_tensor, scene_img_tensor, scene_art_tensor, room_desc_indices, name, index
        
visual_bb_ftsize_k = {'rn18': 512, 'rn34': 512, 'rn50': 2048, 'rn101': 2048, 'vitb16': 768, 'vitb32': 768, 'openclip': 512}

GENERALIST_FEAT_SIZE_DICT = {
    'clip': 512,
    'blip_base': 768,
    'mobile_clip': 512
}

for vn in visual_bb_ftsize_k.keys():
    if os.path.exists(f"preextracted_vectors_wikiart_{vn}/Museum1554-7.unity.pt"):
        print(vn, torch.load(f"preextracted_vectors_wikiart_{vn}/Museum1554-7.unity.pt", weights_only=True).shape)
    else:
        print(vn, "not found")

print("="*20)

for vn in GENERALIST_FEAT_SIZE_DICT.keys():
    print("="*10, vn)
    if vn == "clip":
        print(torch.load(f"./tmp_museums/open_clip_features_museums3k/images/Museum1554-7.unity.pt", weights_only=True).shape, torch.load(f"./tmp_museums/open_clip_features_museums3k/images/Museum1554-7.unity.pt", weights_only=True).dtype)
        print(torch.load(f"./tmp_museums/open_clip_features_museums3k/descriptions/sentences/Museum1554-7.unity.pt", weights_only=True).shape, torch.load(f"./tmp_museums/open_clip_features_museums3k/descriptions/sentences/Museum1554-7.unity.pt", weights_only=True).dtype)
        print(len(pd.read_pickle(f"./tmp_museums/open_clip_features_museums3k/descriptions/tokens_strings/Museum1554-7.unity.pkl")))
    else:
        print(torch.load(f"./tmp_museums/museums3k_new_features/{vn}/images/Museum1554-7.unity.pt", weights_only=True).shape, torch.load(f"./tmp_museums/museums3k_new_features/{vn}/images/Museum1554-7.unity.pt", weights_only=True).dtype)
        print(torch.load(f"./tmp_museums/museums3k_new_features/{vn}/descriptions/sentences/Museum1554-7.unity.pt", weights_only=True).shape, torch.load(f"./tmp_museums/museums3k_new_features/{vn}/descriptions/sentences/Museum1554-7.unity.pt", weights_only=True).dtype)
        print(len(pd.read_pickle(f"./tmp_museums/museums3k_new_features/{vn}/descriptions/tokens_strings/Museum1554-7.unity.pkl")))
        




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
        print("-"*5)
        print("{ds_r1};{ds_r5};{ds_r10};{sd_r1};{sd_r5};{sd_r10};{ds_medr};{sd_medr}")
        print(f"{ds_r1};{ds_r5};{ds_r10};{sd_r1};{sd_r5};{sd_r10};{ds_medr};{sd_medr}")
        print("-"*5)
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
    assert len(avail_models) == 1, avail_models
    model_name_ = avail_models[0]
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
    
    
    
    
def main_proc(_loader, _model_desc_pov, _model_pov, _phase, _indices, _eval, _scheduler, _loss_fn, _loss_fn_room, _optimizer, _room_loss_weight):
    total_loss = 0
    num_batches = 0
    
    output_description_total = torch.empty(len(_indices[_phase]), output_feature_size)
    output_pov_total = torch.empty(len(_indices[_phase]), output_feature_size)
    
    with torch.set_grad_enabled(_phase == 'train'):
        for i, batch_data in enumerate(_loader):
            if _phase == "train":
                if len(batch_data) == 9:
                    data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov, inner_lengths, outer_lengths = batch_data
                elif len(batch_data) == 7:
                    data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov = batch_data
                else:
                    data_desc_pov, data_pov, data_art, names, indexes, len_pov = batch_data
            else:
                if len(batch_data) == 8:
                    data_desc_pov, data_pov, data_art, names, indexes, len_pov, inner_lengths, outer_lengths = batch_data
                elif len(batch_data) == 7:
                    data_desc_pov, data_pov, data_art, raw_descs, names, indexes, len_pov = batch_data
                else:
                    data_desc_pov, data_pov, data_art, names, indexes, len_pov = batch_data
            data_desc_pov = data_desc_pov.to(device)
            data_pov = data_pov.to(device)
            data_art = data_art.to(device)
            # print(inner_lengths, outer_lengths)


            if not _eval:
                _optimizer.zero_grad()

            bsz, fts, no_room_times_no_imgs = data_pov.shape

            if isinstance(_model_desc_pov, HierGRUNet):
                output_room_lev_desc, output_desc_pov = _model_desc_pov(data_desc_pov, inner_lengths, outer_lengths)
                
            else:
                output_room_lev_desc = None
                output_desc_pov = _model_desc_pov(data_desc_pov)


            if isinstance(_model_pov, MyHierBaseline_v3):
                output_pov_img_level, output_pov_room_level, output_pov = _model_pov(data_pov, data_art, len_pov, imgs_per_room=12)
                room_len_pov = len_pov // 12
                tmp_output_pov_room_level = torch.cat([output_pov_room_level[i, :ix] for i, ix in enumerate(room_len_pov)], 0)
                # print(tmp_output_pov_room_level.shape)  # (B*, F)
                
            else:
                tmp_output_pov_room_level = None
                output_pov = _model_pov(data_pov, x_art=data_art if not args.no_artexp else None)

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

            loss_contrastive = _loss_fn.calculate_loss(multiplication_dp)
            if output_room_lev_desc is not None and tmp_output_pov_room_level is not None:
                loss_contrastive_room = _loss_fn_room.calculate_loss(multiplication_dp_room)
                loss_contrastive_total = loss_contrastive + _room_loss_weight*loss_contrastive_room
            else:
                loss_contrastive_total = loss_contrastive


            if not _eval:
                loss_contrastive_total.backward()
                _optimizer.step()

            total_loss += loss_contrastive.item()
            num_batches += 1

            if use_wandb:
                _pps = {
                    f"{_phase}/loss": loss_contrastive.item(), 
                    f"{_phase}/loss_total": loss_contrastive_total.item(), 
                    "scheduler/lr": _scheduler.get_last_lr()[0]
                }
                if output_room_lev_desc is not None:
                    _pps[f"{_phase}/loss_room"] = loss_contrastive_room.item()
                wandb.log()
        
        if _phase == "train":
            _scheduler.step()
            # print(_scheduler.get_last_lr())
        
        epoch_loss = total_loss / num_batches
        if use_wandb:
            wandb.log({
                f"{_phase}/epoch_loss": epoch_loss, 
            })

        # print(f'Loss {_phase.upper()}', epoch_loss)
        _loss_fn.on_epoch_end(epoch_loss, train=_eval)
        
        return output_description_total, output_pov_total

    

    
    
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

    args = parser.parse_args()
    
    
    import wandb
    use_wandb = False
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
    from torch.nn.utils.rnn import pad_packed_sequence
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np
    import time

    import random
    import string
    run_folder = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    batch_size = 64

    visual_backbone = args.artexp
    visual_bb_ftsize = visual_bb_ftsize_k[visual_backbone]
    GENERALIST = args.generalist
    GENERALIST_FEAT_SIZE = GENERALIST_FEAT_SIZE_DICT[GENERALIST]
    if GENERALIST == "clip":
        base_path = "./tmp_museums/open_clip_features_museums3k"
    else:
        base_path = f"./tmp_museums/museums3k_new_features/{GENERALIST}"

    train_dataset = DescriptionSceneMuseum(f"./{base_path}/descriptions/sentences", 
                                           f"./{base_path}/descriptions/tokens_strings", 
                                           f"./{base_path}/images",
                                           f"./preextracted_vectors_wikiart_{visual_backbone}",
                                    indices, "train")

    val_dataset = DescriptionSceneMuseum(f"./{base_path}/descriptions/sentences", 
                                           f"./{base_path}/descriptions/tokens_strings", 
                                           f"./{base_path}/images",
                                           f"./preextracted_vectors_wikiart_{visual_backbone}",
                                    indices, "val")

    test_dataset = DescriptionSceneMuseum(f"./{base_path}/descriptions/sentences", 
                                           f"./{base_path}/descriptions/tokens_strings", 
                                           f"./{base_path}/images",
                                           f"./preextracted_vectors_wikiart_{visual_backbone}",
                                    indices, "test")
    
    collate_fn_to_use = collate_fn if not args.no_hiertxt else collate_fn_desc
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_to_use, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_to_use, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_to_use, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    num_epochs = 50
    number_of_tries = 3
    final_output_strings = []

    output_feature_size = 256 # default: 256
    is_bidirectional = 'mono' not in args.room_txt_agg
    is_bidirectional_RNN_scenes = 'mono' not in args.room_vis_agg
    room_loss_weight = 1.
    room_txt_agg = args.room_txt_agg  # rnn, avg
    room_vis_agg = args.room_vis_agg  # rnn, avg

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

    for n_try in range(number_of_tries):
        lr = 0.001  # default: 0.008

        loss_fn = LossContrastive(approach_name, patience=25, delta=0.0001)
        loss_fn_room = LossContrastive(approach_name, patience=25, delta=0.0001)

        if args.no_hiertxt:
            model_desc_pov = GRUNet(hidden_size=output_feature_size, num_features=GENERALIST_FEAT_SIZE, is_bidirectional=is_bidirectional)
        else:
            model_desc_pov = HierGRUNet(hidden_size=output_feature_size, num_features=GENERALIST_FEAT_SIZE, is_bidirectional=is_bidirectional, room_txt_agg=room_txt_agg)
        
        if args.no_hiervis:
            model_pov = MyBaseline(in_channels=GENERALIST_FEAT_SIZE, out_channels=256, feature_size=output_feature_size, art_features_size=visual_bb_ftsize if not args.no_artexp else 0)
        else:
            model_pov = MyHierBaseline_v3(in_channels=GENERALIST_FEAT_SIZE, out_channels=256, feature_size=output_feature_size, art_features_size=visual_bb_ftsize if not args.no_artexp else 0, bidirectional=is_bidirectional_RNN_scenes, room_vis_agg=room_vis_agg)

        print(model_pov)
        print(model_desc_pov)
        model_desc_pov.to(device)
        model_pov.to(device)

        params = list(model_desc_pov.parameters()) + list(model_pov.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        step_size = 27
        gamma = 0.75
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        sched_name = StepLR

        if use_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="Museums",

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

        best_r10 = 0
        print(f"({n_try+1}/{number_of_tries}) Train procedure ...")
        for ep in tqdm(range(num_epochs)):

            model_desc_pov.train()
            model_pov.train()

            _, _ = main_proc(train_loader, model_desc_pov, model_pov, "train", indices, False, scheduler, loss_fn, loss_fn_room, optimizer, room_loss_weight)

            model_desc_pov.eval()
            model_pov.eval()

            output_description_val, output_pov_val = main_proc(val_loader, model_desc_pov, model_pov, "val", indices, True, scheduler, loss_fn, loss_fn_room, optimizer, room_loss_weight)

            r1, r5, r10, _, _, _, _, _, _, _ = evaluate(output_description=output_description_val,
                                                        output_scene=output_pov_val, section='val',
                                                        out_values=ep % 5 == 4)

            if r10 > best_r10:
                best_r10 = r10
                save_best_model(f"{approach_name}_{n_try}", run_folder, model_pov.state_dict(), model_desc_pov.state_dict())

            if use_wandb:
                wandb.log({
                    "val/T2S_R@1": r1, 
                    "val/T2S_R@5": r5, 
                    "val/T2S_R@10": r10, 
                })

            output_description_val_train, output_pov_val_train = main_proc(train_loader, model_desc_pov, model_pov, "train", indices, True, scheduler, loss_fn, loss_fn_room, optimizer, room_loss_weight)

            r1, r5, r10, _, _, _, _, _, _, _ = evaluate(output_description=output_description_val_train,
                                                        output_scene=output_pov_val_train, section='TRAIN',
                                                        out_values=ep % 5 == 4)
            if use_wandb:
                wandb.log({
                    "train/T2S_R@1": r1, 
                    "train/T2S_R@5": r5, 
                    "train/T2S_R@10": r10, 
                })

        bm_pov, bm_desc_pov = load_best_model(f"{approach_name}_{n_try}", run_folder)
        model_pov.load_state_dict(bm_pov)
        model_desc_pov.load_state_dict(bm_desc_pov)

        model_pov.eval()
        model_desc_pov.eval()
        output_description_test, output_pov_test = main_proc(test_loader, model_desc_pov, model_pov, "test", indices, True, scheduler, loss_fn, loss_fn_room, optimizer, room_loss_weight)

        ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr, formatted_string = evaluate(
            output_description=output_description_test,
            output_scene=output_pov_test,
            section="test",
            out_values=True,
            excel_format=True)

        if use_wandb:
            wandb.log({
                "test/T2S_R@1": ds1, 
                "test/T2S_R@5": ds5, 
                "test/T2S_R@10": ds10,
                "test/S2T_R@1": sd1, 
                "test/S2T_R@5": sd5, 
                "test/S2T_R@10": sd10, 
            })

        final_output_strings.append(formatted_string)

    if use_wandb:
        wandb.finish()
    for out_str in final_output_strings:
        print(out_str)





