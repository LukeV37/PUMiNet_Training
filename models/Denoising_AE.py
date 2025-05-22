import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Encoder, self).__init__()
        self.pre_norm_Q = nn.LayerNorm(embed_dim)
        self.pre_norm_K = nn.LayerNorm(embed_dim)
        self.pre_norm_V = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_heads=num_heads,batch_first=True, dropout=0.25)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim,embed_dim)
    def forward(self, Query, Key, Value):
        Query = self.pre_norm_Q(Query)
        Key = self.pre_norm_K(Key)
        Value = self.pre_norm_V(Value)
        context, weights = self.attention(Query, Key, Value)
        context = self.post_norm(context)
        latent = Query + context
        tmp = F.gelu(self.out(latent))
        latent = latent + tmp
        return latent, weights

class Denoising_AE(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim):
        super(Denoising_AE, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.num_trk_feats = 6

        # Initializer
        self.trk_initializer = nn.Linear(self.num_trk_feats, self.embed_dim)

        # Encode Event
        self.Encode1 = Encoder(self.embed_dim, self.num_heads)
        self.Encode2 = Encoder(self.embed_dim, self.num_heads)
        self.Encode3 = Encoder(self.embed_dim, self.num_heads)

        # AE Compression
        self.Compress1 = nn.Linear(self.embed_dim, self.latent_dim*64)
        self.Compress2 = nn.Linear(self.latent_dim*64, self.latent_dim*32)
        self.Compress3 = nn.Linear(self.latent_dim*32, self.latent_dim*16)
        self.Compress4 = nn.Linear(self.latent_dim*16, self.latent_dim*8)
        self.Compress5 = nn.Linear(self.latent_dim*8, self.latent_dim*4)
        self.Compress6 = nn.Linear(self.latent_dim*4, self.latent_dim*2)
        self.Compress7 = nn.Linear(self.latent_dim*2, self.latent_dim*1)

        # AE Decompression
        self.Decompress1 = nn.Linear(self.latent_dim*1, self.latent_dim*2)
        self.Decompress2 = nn.Linear(self.latent_dim*2, self.latent_dim*4)
        self.Decompress3 = nn.Linear(self.latent_dim*4, self.latent_dim*8)
        self.Decompress4 = nn.Linear(self.latent_dim*8, self.latent_dim*16)
        self.Decompress5 = nn.Linear(self.latent_dim*16, self.latent_dim*32)
        self.Decompress6 = nn.Linear(self.latent_dim*32, self.latent_dim*64)
        self.Decompress7 = nn.Linear(self.latent_dim*64, self.embed_dim)

        # Decode Event
        self.Decode1 = Encoder(self.embed_dim, self.num_heads)
        self.Decode2 = Encoder(self.embed_dim, self.num_heads)
        self.Decode3 = Encoder(self.embed_dim, self.num_heads)

        # Regression Task
        self.regression = nn.Linear(self.embed_dim, self.num_trk_feats)
        
    def forward(self, tracks):
        # Init
        tracks = F.gelu(self.trk_initializer(tracks))

        # Encoder
        tracks, weights = self.Encode1(tracks, tracks, tracks)
        tracks, weights = self.Encode2(tracks, tracks, tracks)
        tracks, weights = self.Encode3(tracks, tracks, tracks)

        # Reduction
        num_tracks = len(tracks)
        tracks = torch.sum(tracks,dim=0)

        # Compression
        tracks = F.gelu(self.Compress1(tracks))
        tracks = F.gelu(self.Compress2(tracks))
        tracks = F.gelu(self.Compress3(tracks))
        tracks = F.gelu(self.Compress4(tracks))
        tracks = F.gelu(self.Compress5(tracks))
        tracks = F.gelu(self.Compress6(tracks))
        tracks = F.gelu(self.Compress7(tracks))

        # Decompression
        tracks = F.gelu(self.Decompress1(tracks))
        tracks = F.gelu(self.Decompress2(tracks))
        tracks = F.gelu(self.Decompress3(tracks))
        tracks = F.gelu(self.Decompress4(tracks))
        tracks = F.gelu(self.Decompress5(tracks))
        tracks = F.gelu(self.Decompress6(tracks))
        tracks = F.gelu(self.Decompress7(tracks))

        # Generation
        tracks = torch.stack([tracks]*num_tracks, dim=0)

        # Decoding
        tracks, weights = self.Decode1(tracks, tracks, tracks)
        tracks, weights = self.Decode2(tracks, tracks, tracks)
        tracks, weights = self.Decode3(tracks, tracks, tracks)

        # Regression
        tracks = self.regression(tracks)

        return tracks
