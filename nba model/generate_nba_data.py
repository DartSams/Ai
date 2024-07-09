from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time

seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2005, 2025)]

all_games = pd.DataFrame()

for season in seasons:
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        games = gamefinder.get_data_frames()[0]
        all_games = pd.concat([all_games, games], ignore_index=True)
        print(f"Fetched data for season {season}")
        time.sleep(1)  
    except Exception as e:
        print(f"Error fetching data for season {season}: {e}")

all_games.to_excel('nba.xlsx', index=False)
print("Data saved to nba.xlsx")
