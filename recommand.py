import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from train import TextCNN
class MovieRecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, title_vocab_size, genres_vocab_size, 
                 title_len=15, genres_len=18, embed_size=32, hidden_size=200):
        super(MovieRecommendationModel, self).__init__()
        
        # 用户特征塔
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.user_fc = nn.Linear(embed_size, hidden_size)
        
        # 电影特征塔
        self.movie_embedding = nn.Embedding(num_movies, embed_size)
        self.movie_fc = nn.Linear(embed_size, hidden_size)  # 新增层

        # 标题特征提取
        self.title_cnn = TextCNN(
            vocab_size=title_vocab_size,
            embed_size=embed_size,
            num_filters=32,
            filter_sizes=[3, 4, 5]
        )
        self.title_fc = nn.Linear(96, hidden_size)  # 3 filter sizes * 32 filters
        
        # 类型特征提取
        self.genres_embedding = nn.Embedding(genres_vocab_size, embed_size)
        self.genres_fc = nn.Linear(embed_size, hidden_size)
        
        # 预测层
        self.predict_fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1))
        
        self.predict_fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),  # 新增dropout
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1))

    def forward(self, user_id, movie_id, title, genres):
        # 用户特征
        user_embed = self.user_embedding(user_id)
        user_feature = self.user_fc(user_embed)
        
        # 电影特征
        movie_embed = self.movie_embedding(movie_id)
        movie_embed = self.movie_fc(movie_embed)

        # 标题特征
        if title.dim() == 1:
           title = title.unsqueeze(1)
        title_feature = self.title_cnn(title)
        title_feature = self.title_fc(title_feature)
        
        # 类型特征
        genres_embed = self.genres_embedding(genres)
        genres_embed = genres_embed.view(genres_embed.size(0), -1)  # 展平
        genres_feature = self.genres_fc(genres_embed)
        if genres_embed.dim() == 2:  # 如果已经是[batch, embed_size]
           genres_embed = genres_embed.unsqueeze(1)  # 变为[batch, 1, embed_size]
        

        # 合并电影特征
        movie_feature = movie_embed + title_feature + genres_feature
        
        # 预测评分
        combined = torch.cat([user_feature, movie_feature], dim=1)
        rating = self.predict_fc(combined)
        

        return rating.squeeze()