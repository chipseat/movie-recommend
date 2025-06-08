import torch.nn as nn
import torch.nn.functional as F
import torch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,  # 明确指定输入通道数
                     out_channels=num_filters, 
                     kernel_size=(k, embed_size)) 
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_size]
        x = x.unsqueeze(1)    # [batch_size, 1, seq_len, embed_size]
        
        # 通过多个卷积层并应用ReLU激活
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # 对每个卷积结果进行最大池化
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        x = torch.cat(x, 1)   # 拼接所有卷积特征
        x = self.dropout(x)
        return x