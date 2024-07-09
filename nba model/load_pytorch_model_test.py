import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib  

class Model(nn.Module):
    def __init__(self, features=3, h1=8, h2=9, out_features=2): 
        super().__init__()
        self.fc1 = nn.Linear(features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x


model = Model()
model.load_state_dict(torch.load("nba_model.pt"))

scaler = joblib.load('scaler.joblib')

team_mapping = {'MIA': 0, 'DAL': 1, 'DET': 2, 'PHX': 3, 'SAS': 4, 'CLE': 5, 'LAC': 6, 'NJN': 7, 'CHI': 8, 'MEM': 9, 'MIL': 10, 'LAL': 11, 'SAC': 12, 'WAS': 13, 'IND': 14, 'NYK': 15, 'HOU': 16, 'BOS': 17, 'POR': 18, 'TOR': 19, 'MIN': 20, 'SEA': 21, 'GSW': 22, 'UTA': 23, 'ATL': 24, 'ORL': 25, 'NOK': 26, 'ABQ': 27, 'FTW': 28, 'NYK': 29, 'UTA': 30, 'MIN': 31, 'DEN': 32}
matchup_mapping = {'DAL': 0, 'MIA': 1, 'DET': 2, 'PHX': 3, 'SAS': 4, 'CLE': 5, 'LAC': 6, 'NJN': 7, 'CHI': 8, 'MEM': 9, 'MIL': 10, 'LAL': 11, 'SAC': 12, 'WAS': 13, 'IND': 14, 'NYK': 15, 'HOU': 16, 'BOS': 17, 'POR': 18, 'TOR': 19, 'MIN': 20, 'SEA': 21, 'GSW': 22, 'UTA': 23, 'ATL': 24, 'ORL': 25, 'NOK': 26, 'ABQ': 27, 'FTW': 28, 'NYK': 29, 'UTA': 30, 'MIN': 31, 'DEN': 32}


new_data_tensor = torch.FloatTensor([0,4,1])

model.eval()
with torch.no_grad():
    pred = model(new_data_tensor)
    print(pred)
    predicted_class = pred.argmax().item()
    print(f'Predicted class: {predicted_class}')
