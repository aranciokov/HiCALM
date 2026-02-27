import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class MVCNN(nn.Module):
    def __init__(self, in_channels, out_channels, feature_size=4096, art_features_size=2048):
        super().__init__()
        self.use_art = art_features_size != 0
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels+art_features_size if self.use_art else in_channels, feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(feature_size, feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, out_channels),
        )


    def forward(self, x, x_art, list_length=None, clip_mask=None, imgs_per_room=None):
        # took inspiration from: https://github.com/RBirkeland/MVCNN-PyTorch/blob/master/models/mvcnn.py
        # we receive x of shape (bsz, in_channels, n_views)
        x = x.to(torch.float32)

        if self.use_art:
            x = torch.cat((
                x.transpose(1, 2), 
                x_art.transpose(1, 2)
            ), -1)
        else:
            x = x.transpose(1, 2) # (bsz, n_views, in_channels)

        lengths = [x.shape[1]] * x.shape[0]
        if list_length is not None:
            assert len(list_length) == x.shape[0]
            lengths = list_length
        
        all_pooled_views = []
        for m, museum in enumerate(x):
            pooled_view = museum[0]
            for i in range(1, lengths[m]):
                pooled_view = torch.max(pooled_view, museum[i])
            all_pooled_views.append(pooled_view)
        pooled_view = torch.stack(all_pooled_views, 0)  # (bsz, in_channels [+ art_features_size])
        
        pooled_view = self.classifier(pooled_view)
        return None, None, pooled_view
    

class MVCNNSA(nn.Module):
    def __init__(self, in_channels, out_channels, feature_size=4096, art_features_size=2048):
        super().__init__()
        self.use_art = art_features_size != 0
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels+art_features_size if self.use_art else in_channels, feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(feature_size, out_channels),
        )
        self.V = nn.Sequential(
            nn.Linear(in_channels+art_features_size if self.use_art else in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.ReLU(inplace=True),
        )


    def forward(self, x, x_art, list_length=None, clip_mask=None, imgs_per_room=None):
        # took inspiration from: https://github.com/vaibhavnayel/MVCNN-SA/blob/master/mvcnn_att.py
        # we receive x of shape (bsz, in_channels, n_views)
        x = x.to(torch.float32)

        if self.use_art:
            x = torch.cat((
                x.transpose(1, 2), 
                x_art.transpose(1, 2)
            ), -1)
        else:
            x = x.transpose(1, 2) # (bsz, n_views, in_channels)

        lengths = [x.shape[1]] * x.shape[0]
        if list_length is not None:
            assert len(list_length) == x.shape[0]
            lengths = list_length
        
        all_pooled_views = []
        for m, museum in enumerate(x):
            filt_museums = museum[:lengths[m]]  # (n_views, in_channels [+ art_features_size])
            weights = torch.softmax(self.V(filt_museums), 0)
            pooled_view = (weights * filt_museums).sum(0)
            all_pooled_views.append(pooled_view)
        pooled_view = torch.stack(all_pooled_views, 0)  # (bsz, in_channels [+ art_features_size])
        
        pooled_view = self.classifier(pooled_view)
        return None, None, pooled_view


from dan import DAN
from vsformer import VSFormer
    

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



if __name__ == "__main__":
    # simple test
    x = torch.randn(2, 512, 10)
    x_art = torch.randn(2, 2048, 10)
    list_length = [10, 7]
    clip_mask = torch.ones(2, 10, 1)
    
    model = MyHierBaseline_v3(in_channels=512, out_channels=256, feature_size=128, art_features_size=2048, bidirectional=True, room_vis_agg="rnn")
    x_img, x_room, x_museum = model(x, x_art, list_length=list_length, clip_mask=clip_mask, imgs_per_room=5)
    print("Hier")
    print(x_img.shape)      # (2, 10, 256)
    print(x_room.shape)     # (2, 2, 256)
    print(x_museum.shape)   # (2, 256) or (2, 256*2) if bidirectional
    
    print("MVCNN")
    model = MVCNN(in_channels=512, out_channels=256, feature_size=128, art_features_size=2048)
    _, _, x_museum = model(x, x_art, list_length=list_length, clip_mask=clip_mask, imgs_per_room=5)
    print(x_museum.shape)   # (2, 256) or (2, 256*2) if bidirectional
    
    print("MVCNNsa")
    model = MVCNNSA(in_channels=512, out_channels=256, feature_size=128, art_features_size=2048)
    _, _, x_museum = model(x, x_art, list_length=list_length, clip_mask=clip_mask, imgs_per_room=5)
    print(x_museum.shape)   # (2, 256) or (2, 256*2) if bidirectional
    
    print("DAN")
    model = DAN(h=4, feature_dim=512, num_heads=8, inner_dim=128, dropout=0.1, art_features_size=2048)
    x_museum, _ = model(batch_size=x.shape[0], max_num_views=x.shape[2], num_views=list_length, x=torch.cat((
                x.transpose(1, 2), 
                x_art.transpose(1, 2)
            ), -1))
    
    print(x_museum.shape)   # (2, 256) or (2, 256*2) if bidirectional

    print("VSFormer")
    model = VSFormer(feature_dim=512, num_layers=4, num_heads=8, attention_dropout=0.1, mlp_dropout=0.1, widening_factor=2, art_features_size=2048)
    x_museum, _ = model(batch_size=x.shape[0], max_num_views=x.shape[2], num_views=list_length, x=torch.cat((
                x.transpose(1, 2), 
                x_art.transpose(1, 2)
            ), -1))
    
    print(x_museum.shape)   # (2, 256) or (2, 256*2) if bidirectional
