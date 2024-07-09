import random
import numpy as np


teams = ["MIA","BOS","DAL","76ERS","MIL"]

# print(f"{random.choice(teams)} vs {random.choice(teams)}")

x={
    "team":random.choice(teams),
    "opponent":random.choice(teams),
    "full rostor":1,
    "state":"away",
    "outcome":"W"
}

# print(x)

team_names = ["Team"]
opponent_team = ["Opponenet team"]
rostor = ["full rostor"]
state = ["state"]
outcome = ["outcome"]


for i in range(200):
    team_names.append(random.choice(teams))
    opponent_team.append(random.choice(teams))
    rostor.append(random.randint(0.0,1.0))
    state.append(random.choice(["home","away"]))
    outcome.append(random.choice(["W","L"]))


np.savetxt('scores.csv', [p for p in zip(team_names, opponent_team, rostor, state,outcome)], delimiter=',', fmt='%s')

