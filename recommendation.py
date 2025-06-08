import torch
from preprocess import users
def generate_recommendations(model, user_id, movies_df, device, top_n=10):
    model.eval()
    model.to(device)
    
    # 准备用户数据
    user_data = users[users['UserID'] == user_id].iloc[0]
    user_id_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
    
    # 准备所有电影数据
    movie_ids = movies_df['MovieID'].values
    titles = [x[0] if isinstance(x, list) else 0 for x in movies_df['Title']]  # 取列表第一个元素或默认0
    genres = [x[0] if isinstance(x, list) else 0 for x in movies_df['Genres']]  # 同上
    
    # 转换为张量
    movie_ids_tensor = torch.tensor(movie_ids, dtype=torch.long).to(device)
    titles_tensor = torch.tensor(titles, dtype=torch.long).to(device)
    genres_tensor = torch.tensor(genres, dtype=torch.long).to(device)
    
    # 预测评分
    with torch.no_grad():
        pred_ratings = model(
            user_id_tensor.repeat(len(movie_ids)),
            movie_ids_tensor,
            titles_tensor,
            genres_tensor
        )
    
    # 获取Top-N推荐
    _, top_indices = torch.topk(pred_ratings, top_n)
    top_movie_ids = movie_ids[top_indices.cpu().numpy()]
    
    # 返回推荐电影信息
    recommendations = movies_df[movies_df['MovieID'].isin(top_movie_ids)]
    return recommendations[['MovieID', 'Title', 'Genres']]