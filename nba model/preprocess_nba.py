import pandas as pd

def get_opponent_team(matchup):
    if "@" in matchup:
        return matchup.split(" @ ")[1]
    else:
        return matchup.split(" vs. ")[1]

input_file = "nba.xlsx"  
df = pd.read_excel(input_file)

df["MATCHUP"] = df["MATCHUP"].apply(get_opponent_team)

output_file = "modified_nba_dataset.xlsx"
df.to_excel(output_file, index=False)

print("The MATCHUP column has been updated and saved to", output_file)





input_file = 'scores.csv' 
output_file = 'scores.csv'  

with open(input_file, 'r') as file:
    file_data = file.read()

file_data = file_data.replace('\t', ',')

with open(output_file, 'w') as file:
    file.write(file_data)

print(f"Tabs have been replaced with commas and saved to {output_file}")