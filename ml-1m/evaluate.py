import torch
import torch.nn.functional as F
def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)
    
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            title = batch['title'].to(device)
            genres = batch['genres'].to(device)
            target = batch['target'].to(device)
            
            outputs = model(user_id, movie_id, title, genres)
            loss = F.mse_loss(outputs, target)
            total_loss += loss.item()
    
    rmse = torch.sqrt(torch.tensor(total_loss / len(dataloader)))
    print(f'Test RMSE: {rmse:.4f}')
    return rmse