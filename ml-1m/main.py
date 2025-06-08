from model import MovieDataset
from recommand import MovieRecommendationModel
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import torch.nn as nn
from train1 import train_model
from evaluate import evaluate_model
from recommendation import generate_recommendations
# 在main.py开头添加

def main():
    # 加载预处理数据
    (title_count, title_set, genres2int, features, targets_values, 
     ratings, users, movies, data, movies_orig, users_orig) = pickle.load(open('preprocess.p', 'rb'))
    
    # 准备数据集
    dataset = MovieDataset(features, targets_values)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    num_users = ratings['UserID'].max() + 1  # 最大ID+1
    num_movies = ratings['MovieID'].max() + 1
    title_vocab_size = len(title_set)
    genres_vocab_size = len(genres2int)
    
    model = MovieRecommendationModel(
        num_users=num_users,
        num_movies=num_movies,
        title_vocab_size=title_vocab_size,
        genres_vocab_size=genres_vocab_size
    )
    
    # 训练配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    model = train_model(model, train_loader, criterion, optimizer, device)
    
    # 评估模型
    evaluate_model(model, test_loader, device)
    
    # 生成推荐
    user_id = 1  # 示例用户ID
    recommendations = generate_recommendations(model, user_id, movies, device)
    print("\nTop 10 Recommendations for User", user_id)
    print(recommendations)
    
    # 保存模型
    torch.save(model.state_dict(), 'movie_recommendation_model.pth')

if __name__ == '__main__':
    main()