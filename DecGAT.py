import torch
from torch import nn
import torch.nn.functional as F
import dgl
from sparse_encoder import SparseFeatureEncoder
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.nn import GraphConv
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DecGAT(nn.Module):
    def __init__(self, feature_dim, input_dim,  output_dim):
        super(DecGAT, self).__init__()
        self.etypes = ['cor','sim']
        self.dropout = 0.1
        self.layer1_one = nn.ModuleDict({mode: Relationship_Decoupled_layer(input_dim, dropout=self.dropout) for mode in self.etypes})
        self.layer1_two = nn.ModuleDict({mode: One_Neighbor_Aggregation(input_dim, dropout=self.dropout) for mode in self.etypes})
        self.layer_three = nn.ModuleDict({mode: Neighbor_Interaction_Aggregation(dropout=self.dropout) for mode in self.etypes})

        self.layer2_one = nn.ModuleDict({mode: Relationship_Decoupled_layer(input_dim, dropout=self.dropout) for mode in self.etypes})
        self.layer2_two = nn.ModuleDict({mode: One_Neighbor_Aggregation(input_dim, dropout=self.dropout) for mode in self.etypes})
        
        self.fc_output = nn.ModuleDict({mode: nn.Linear(input_dim, output_dim) for mode in self.etypes})
        self.fc_input = nn.ModuleDict({mode: nn.Linear(feature_dim, input_dim) for mode in self.etypes})
        self.reset_parameters()
    def reset_parameters(self):
        pass


    def forward(self, blocks):
            feats = blocks[0].srcdata['feature']
            emb = {mode: feats.clone().detach() for mode in self.etypes}
            for mode in self.etypes:
                emb[mode] = self.fc_input[mode](emb[mode])
                node_alpha = self.layer1_one[mode](blocks[0],emb[mode],mode)
                one_hop_emb = self.layer1_two[mode](blocks[0],emb[mode],node_alpha,mode)
                neighbor_int_emb = self.layer_three[mode](blocks[0],emb[mode],node_alpha, mode)
                emb[mode] = (one_hop_emb + neighbor_int_emb)/2
            for mode in self.etypes:
                node_alpha = self.layer2_one[mode](blocks[1],emb[mode],mode)
                one_hop_emb = self.layer2_two[mode](blocks[1],emb[mode],node_alpha,mode)
                neighbor_int_emb = self.layer_three[mode](blocks[1],emb[mode],node_alpha, mode)
                emb[mode] = (one_hop_emb + neighbor_int_emb)/2
                emb[mode] = self.fc_output[mode](emb[mode])
            return emb


class Relationship_Decoupled_layer(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Relationship_Decoupled_layer, self).__init__()
        self.etypes = ['cor','sim']
        self.feat_drop = nn.Dropout(dropout)
        self.node_attention = nn.Linear(input_dim, 2, bias=False)
        self.e_edge_attention = nn.Linear(2*input_dim, 2, bias=False)
        self.i_edge_attention = nn.Linear(2*input_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.node_attention.weight, gain=gain)
        nn.init.xavier_normal_(self.e_edge_attention.weight, gain=gain)
        nn.init.xavier_normal_(self.i_edge_attention.weight, gain=gain)

    def e_attention(self, edges):
        z = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        edge_alpha = self.e_edge_attention(z)
        return {'e': F.softmax(edge_alpha,dim=1)}
    
    def i_attention(self, edges):
        z = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        edge_alpha = self.i_edge_attention(z)
        return {'e': torch.sigmoid(edge_alpha)}

    def forward(self, graph, feat, mode): 
        graph.srcdata['z']= self.feat_drop(feat)
        graph.dstdata['z'] = graph.srcdata['z'][:graph.number_of_dst_nodes()]
        node_alpha = self.node_attention(graph.dstdata['z'])
        node_alpha = F.softmax(node_alpha,dim=1)
        for type in self.etypes:
            if type == mode:
                graph.apply_edges(self.e_attention, etype = type)
            else:
                graph.apply_edges(self.i_attention, etype = type)
        return node_alpha


class One_Neighbor_Aggregation(nn.Module):
    def __init__(self, input_dim, dropout):
        super(One_Neighbor_Aggregation, self).__init__()
        self.feat_drop = nn.Dropout(dropout)
        self.Weight = nn.Linear(input_dim,input_dim)
        self.reset_parameters()

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e':edges.data['e'][:,0]}
    def reduce_func(self, nodes):
        f = torch.mean(nodes.mailbox['e'].unsqueeze(2)* self.Weight(nodes.mailbox['z']), dim=1)
        return {'f': F.relu(f)}

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.Weight.weight, gain=gain)
    
    def forward(self, graph, feat, node_alpha, mode): 
        feat_src = self.feat_drop(feat) 
        feat_dst = node_alpha[:,0].unsqueeze(1)*feat_src[:graph.number_of_dst_nodes()] 
        graph.srcdata['z']= feat_src
        graph.dstdata['z'] = feat_dst
        graph.multi_update_all({mode:(self.message_func, self.reduce_func)},'sum')
        return graph.dstdata['f']+graph.dstdata['z']

class Neighbor_Interaction_Aggregation(nn.Module):
    def __init__(self, dropout):
        super(Neighbor_Interaction_Aggregation, self).__init__()
        self.feat_drop = nn.Dropout(dropout)
    def e_message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e'][:,1]}
    def e_reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['e'].unsqueeze(2)* nodes.mailbox['z'], dim=1)
        return {'e_h': h}
    def i_message_func(self, edges):
        return {'z': edges.src['z'], 'e':edges.data['e']}
    def i_reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['e']* nodes.mailbox['z'], dim=1)
        return {'i_h': h}
    def forward(self, graph, feat, self_alpha, mode): 
        feat_src = self.feat_drop(feat)
        feat_dst = self_alpha[:,1].unsqueeze(1)*feat_src[:graph.number_of_dst_nodes()]
        graph.dstdata['z'] = feat_dst
        graph.srcdata['z'] = feat_src
        if mode == 'sim':
            graph.multi_update_all({'sim':(self.e_message_func, self.e_reduce_func),'cor':(self.i_message_func, self.i_reduce_func)},'sum')
        else:
            graph.multi_update_all({'cor':(self.e_message_func, self.e_reduce_func),'sim':(self.i_message_func, self.i_reduce_func)},'sum')
        emb =  F.relu(0.5*(torch.square(graph.dstdata['e_h']+graph.dstdata['i_h'])-torch.square(graph.dstdata['e_h'])-torch.square(graph.dstdata['i_h'])))+graph.dstdata['z']
        return emb
