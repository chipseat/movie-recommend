def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            # 移动数据到设备
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            title = batch['title'].to(device)
            genres = batch['genres'].to(device)
            target = batch['target'].to(device)
            
            # 前向传播
            outputs = model(user_id, movie_id, title, genres)
            loss = criterion(outputs, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model