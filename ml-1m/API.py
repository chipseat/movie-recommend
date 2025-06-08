from flask import Flask, request, jsonify
import torch
import pickle
from recommand import MovieRecommendationModel
from recommendation import generate_recommendations

app = Flask(__name__)

# 加载模型和预处理数据
def load_resources():
    global model, movies_df, title_set, genres2int, device, users_df
    
    # 加载预处理数据
    (title_count, title_set, genres2int, _, _, 
     _, _, movies_df, _, _, _) = pickle.load(open('preprocess.p', 'rb'))
    
    # 初始化模型
    num_users = len(users_df['UserID'].unique())
    num_movies = len(movies_df['MovieID'].unique())
    title_vocab_size = len(title_set)
    genres_vocab_size = len(genres2int)
    
    model = MovieRecommendationModel(
        num_users=num_users,
        num_movies=num_movies,
        title_vocab_size=title_vocab_size,
        genres_vocab_size=genres_vocab_size
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load('movie_recommendation_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = data['user_id']
        top_n = data.get('top_n', 10)
        
        recommendations = generate_recommendations(model, user_id, movies_df, device, top_n)
        return jsonify({
            'status': 'success',
            'recommendations': recommendations.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    load_resources()
    app.run(host='0.0.0.0', port=5000)