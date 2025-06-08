from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cold_start_recommendation(new_movie_title, new_movie_genres, top_n=10):
    # 处理新电影标题
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title = pattern.match(new_movie_title).group(1).strip() if pattern.match(new_movie_title) else new_movie_title
    
    # 处理新电影类型
    genres = new_movie_genres.split('|')
    
    # 使用TF-IDF计算标题相似度
    all_titles = [' '.join(movies['Title'].map(lambda x: ' '.join([str(i) for i in x]))) for _, movies in movies.iterrows()]
    all_titles.append(' '.join([str(title2int.get(word, title2int['<PAD>'])) for word in title.split()]))
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(all_titles)
    
    # 计算相似度
    cosine_sim = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 获取相似电影的索引
    movie_indices = [i[0] for i in sim_scores[:top_n]]
    
    # 返回推荐电影
    return movies.iloc[movie_indices][['MovieID', 'Title', 'Genres']]