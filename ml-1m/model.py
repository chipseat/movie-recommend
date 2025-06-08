import torch
from torch.utils.data import Dataset, DataLoader


class MovieDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 解析特征
        user_id = self.features[idx][0]
        gender = self.features[idx][1]
        age = self.features[idx][2]
        job = self.features[idx][3]
        movie_id = self.features[idx][4]
        title = self.features[idx][5][0]
        genres = self.features[idx][6][0]
        if isinstance(title, list):
            title = title[:15] + [0]*(15-len(title))  # 截断或填充到15
        else:
            title = [title] + [0]*14
        if isinstance(genres, list):
            genres = genres[:18] + [0]*(18-len(genres))  # 截断或填充到18
        else:
            genres = [genres] + [0]*17
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'gender': torch.tensor(gender, dtype=torch.long),
            'age': torch.tensor(age, dtype=torch.long),
            'job': torch.tensor(job, dtype=torch.long),
            'movie_id': torch.tensor(movie_id, dtype=torch.long),
            'title': torch.tensor(title, dtype=torch.long),
            'genres': torch.tensor(genres, dtype=torch.long),
            'target': torch.tensor(self.targets[idx][0], dtype=torch.float)
        }
    if __name__ == "__main__":
     print("MovieDataset类存在！")