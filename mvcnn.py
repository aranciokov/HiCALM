# adapted from https://github.com/SMNUResearch/MVP-N/tree/main/model
import torch
import torch.nn as nn

class MVCNN(nn.Module):
    #def __init__(self, model, feature_dim, num_layers, num_heads, attention_dropout, mlp_dropout, widening_factor):
    def __init__(self, feature_dim, output_feature_size, art_features_size=0):
        super(MVCNN, self).__init__()

        #self.extractor = nn.Sequential(*list(model.net.children())[:-1])
        #self.classifier = model.net.fc
        v_feature_dim = feature_dim
        if art_features_size != 0:
            v_feature_dim += art_features_size
        self.classifier = nn.Linear(v_feature_dim, output_feature_size)

    def forward(self, batch_size, max_num_views, num_views, x, use_utilization=False):
        #y = self.extractor(x)
        y = x
        k = []
        u = []
        count = 0
        for i in range(0, batch_size):
            #z = y[:, count:(count + max_num_views), :]
            count += num_views[i]
            z = y[i, 0:num_views[i], :]
            z = z.view(z.shape[0], -1)

            if use_utilization:
                utilization = torch.zeros(max_num_views)
                view_utilization = torch.max(z, 0)[1]
                for j in range(0, z.shape[1]):
                    utilization[view_utilization[j]] += 1

                utilization = utilization / torch.sum(utilization)
                u.append(utilization)

            z = torch.max(z, 0)[0]
            k.append(z)

        k = torch.stack(k) # batch_size * num_features

        if not use_utilization:
            return self.classifier(k), k
        else:
            u = torch.stack(u)

            return self.classifier(k), k, u

