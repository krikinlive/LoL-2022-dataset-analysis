# Leauge Of Legends 2022 Data Set Analysis - How Can Mid/Early Game Features and Positions of The Players Help Us Predicit Their Final Kills
by: Roman Lysenko; email: rlysenko@ucsd.edu
## Introduction
In this project of mine I went over dataset that is publically available by oracleselixir(oracleselixir.com) one league of legends games. The data set provides vast information on player's kills, assists, the position in each uniquly id games/tournaments. The team they are in and so on. However my main goal was to understand how much is it possible, using linear regression model, to predict player's final number of kills using only early-mid game features/information and their in-game position. I understand that league of legends is heavily team base game and so information of team's perfomance is alos can be quite influental towards player's perfomances, however I wanted to see how much is it possible to predicit using only individual features.
## Data Cleaning and Exploratory Data Analysis Part: 1
First I started filtering the dataframe with rows/columns that only correspond to valid individual player gaems, such as a column position has value position = team which resposnds to a row that contains information on the team that is participating in a certain game. I do not need that for my case thus I drop it and stick with these list of columns:
```python
player_cols = [
    # --- IDs / context ---
    "gameid",
    "date",
    "league",
    "year",
    "split",
    "playoffs",
    "patch",
    "side",        # Blue / Red
    "position",    # top / jng / mid / bot / sup
    "teamname",
    "playerid",
    "playername",

    # --- overall outcome for that player ---
    "gamelength",
    "kills",       # <— response variable for regression

    # --- early-game stats at 10 minutes (features) ---
    "goldat10",
    "xpat10",
    "csat10",
    "killsat10",
    "assistsat10",
    "deathsat10",

    # --- optional: early-game stats at 15 minutes (extra features/EDA) ---
    "goldat15",
    "xpat15",
    "csat15",
    "killsat15",
    "assistsat15",
    "deathsat15",
]
```
```
study_df = lol_df.loc[lol_df["position"] != "team", player_cols].head() #Get rid of column teams so we do not deal with team rows
study_df
```
| gameid                | date                | league   |   year | split   |   playoffs |   patch | side   | position   | teamname          | playerid                                  | playername   |   gamelength |   kills |   goldat10 |   xpat10 |   csat10 |   killsat10 |   assistsat10 |   deathsat10 |   goldat15 |   xpat15 |   csat15 |   killsat15 |   assistsat15 |   deathsat15 |
|:----------------------|:--------------------|:---------|-------:|:--------|-----------:|--------:|:-------|:-----------|:------------------|:------------------------------------------|:-------------|-------------:|--------:|-----------:|---------:|---------:|------------:|--------------:|-------------:|-----------:|---------:|---------:|------------:|--------------:|-------------:|
| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | top        | BRION Challengers | oe:player:38e0af7278d6769d0c81d7c4b47ac1e | Soboro       |         1713 |       2 |       3228 |     4909 |       89 |           0 |             0 |            0 |       5025 |     7560 |      135 |           0 |             1 |            0 |
| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | jng        | BRION Challengers | oe:player:637ed20b1e41be1c51bd1a4cb211357 | Raptor       |         1713 |       2 |       3429 |     3484 |       58 |           1 |             2 |            0 |       5366 |     5320 |       89 |           2 |             3 |            2 |
| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | mid        | BRION Challengers | oe:player:d1ae0e2f9f3ac1e0e0cdcb86504ca77 | Feisty       |         1713 |       2 |       3283 |     4556 |       81 |           0 |             1 |            0 |       5118 |     6942 |      120 |           0 |             3 |            0 |
| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | bot        | BRION Challengers | oe:player:998b3e49b01ecc41eacc392477a98cf | Gamin        |         1713 |       2 |       3600 |     3103 |       78 |           1 |             1 |            0 |       5461 |     4591 |      115 |           2 |             1 |            2 |
| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | sup        | BRION Challengers | oe:player:e9741b3a238723ea6380ef2113fae63 | Loopy        |         1713 |       1 |       2678 |     2161 |       16 |           1 |             1 |            0 |       3836 |     3588 |       28 |           1 |             2 |            2 |

Another column that I constructed for myself was game_minutes to have a cleaner look on games length distribution:
```
study_df['game_minutes'] = study_df['gamelength'] / 60
study_df['game_minutes']
```
|   game_minutes |
|---------------:|
|          28.55 |
|          28.55 |
|          28.55 |
|          28.55 |
|          28.55 |

Next I checked that all numeric columns are numeric types and how much data it is missing
```
#Check if these columns are numeric as they should be and also discover which one has missing values
num_cols = [
    # target
    "kills",
    
    # game duration (for EDA, not as a feature at time-of-prediction)
    "gamelength",

    # 10-minute early-game stats (baseline features)
    "goldat10",
    "xpat10",
    "csat10",
    "killsat10",
    "assistsat10",
    "deathsat10",

    # 15-minute stats (optional extra features / EDA)
    "goldat15",
    "xpat15",
    "csat15",
    "killsat15",
    "assistsat15",
    "deathsat15",
    #
    "year",
    "playoffs"
]
for col in num_cols:
    print(col + ": ", study_df[col].dtype, study_df[col].isna().sum())
```
And this is the result I got:
```text
kills:  int64 0
gamelength:  int64 0
goldat10:  float64 18930
xpat10:  float64 18930
csat10:  float64 18930
killsat10:  float64 18930
assistsat10:  float64 18930
deathsat10:  float64 18930
goldat15:  float64 18930
xpat15:  float64 18930
csat15:  float64 18930
killsat15:  float64 18930
assistsat15:  float64 18930
deathsat15:  float64 18930
year:  int64 0
playoffs:  int64 0
```

Then I did the same for categorical columns:
```
cat_cols = [
    "side",      # 'Blue' / 'Red'
    "position",  # 'top', 'jng', 'mid', 'bot', 'sup'
    "league",
    "split",
    "teamname",
    "playerid",
    "playername",
]
for col in cat_cols:
    print(col + ": ", study_df[col].dtype, study_df[col].isna().sum())
```
```text
side:  object 0
position:  object 0
league:  object 0
split:  object 34910
teamname:  object 45
playerid:  object 2209
playername:  object 16
```

## Assessment of Missingness
Before looking at the graphs and looking at the distribution I first decided to asses the missigness because I think it is important to understand what is missing and what is going on with data before looking at the graphs, trying to find patterns and making any kind of analysis. I firstly splitted and created two lists one for numeric columns and one for categorical
```
cat_cols_na = [
    "playerid",
    "playername"
]
num_cols_na = [
    "goldat10",
    "xpat10",
    "csat10",
    "killsat10",
    "assistsat10", 
    "deathsat10",
    "goldat15",
    "xpat15",
    "csat15",
    "killsat15",
    "assistsat15",
    "deathsat15",
    
]
```
#### Missgness for Numeric Columns
I then procceded to check my first assumption which is whether the missgness was somehow related to the position of players:
```
for col in num_cols_na:
    print(col)
    print(study_df.groupby("position")[col].apply(lambda s: s.isna().mean()))
```
```text
goldat10
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: goldat10, dtype: float64
xpat10
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: xpat10, dtype: float64
csat10
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: csat10, dtype: float64
killsat10
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: killsat10, dtype: float64
assistsat10
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: assistsat10, dtype: float64
deathsat10
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: deathsat10, dtype: float64
goldat15
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: goldat15, dtype: float64
xpat15
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: xpat15, dtype: float64
csat15
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: csat15, dtype: float64
killsat15
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: killsat15, dtype: float64
assistsat15
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: assistsat15, dtype: float64
deathsat15
position
bot    0.15
jng    0.15
mid    0.15
sup    0.15
top    0.15
Name: deathsat15, dtype: float64
```
Based on this I can see that all positions have same distribution of missgness among all numeric features which gave an answer that no, missgness was not related to player's position. Also another thing that this showed was that the numeric values seem to have same distribution of missgness in them, which tells us that if we find the reason why one of the columns have it is potentially can explain why others have it too.
Another theory that I decided to test was whether it could be due to games have no recordings for games under 10 minutes, however that also proved wrong
```
print(study_df["game_minutes"].describe().min()) # To check what min values are, now I want to see the games that are under 10 min
print((study_df["game_minutes"] < 10.0).sum())
```
```text
5.646077060014566
0
```
So my next step was to go back to the whole csv file and just look over it myself, which proved to be the most useful. Something that I immidiatly noticed was how a lot of tournmanets, or in case of datasets leagues, sometimes simply had no information recorded on teams players except their wins, game length. The reason why it seemed obvious to me why it is happening is because of my prior experience with gaming community. I'm part of biggest gaming collegiate gaming org in the U.S called Triton Gaming that is based at UCSD. I'm part of the broadcast team and one of the things that I noticed in these small college league tournaments is how little there is information on the games themselves, simply because college students dont have either resources or just human power to record such stats, thus my next thought proccess was simply check which leagues had missgness in these numeric columns, and since all numeric values have same distribution of missgness then by finding one I can explain others.
```
study_df["goldat10_missing"] = study_df["goldat10"].isna()
league_cols = study_df.groupby("league")["goldat10_missing"].mean().sort_values(ascending=False)
list_of_league_na = list((league_cols == 1.0).index)
list_of_league_na
```
what this peace of code will show is the league/tournaments which had 100% missgness for goldat10 column and the result was quite insightful
```
['ASCI',
 'DCup',
 'LPL',
 'LDL',
 'WLDs',
 'CT',
 'PGC',
 'LLA',
 'LMF',
 'CBLOLA',
 'LPLOL',
 'LVP SL',
 'MSI',
 'NEXO',
 'NLC',
 'NLC Aurora Open',
 'PCS',
 'PGN',
 'DDH',
 'PRM',
 'PRMP',
 'SL (LATAM)',
 'TAL',
 'TCL',
 'UL',
 'UPL',
 'USP',
 'VCS',
 'VL',
 'LJLA',
 'LJL',
 'LHE',
 'LFL2',
 'EBL',
 'EBLPA',
 'EL',
 'ESLOL',
 'EUM',
 'GL',
 'GLL',
 'GLLPA',
 'HC',
 'HM',
 'IC',
 'LAS',
 'LCK',
 'LCKC',
 'LCL',
 'LCO',
 'LCS',
 'LCSA',
 'CDF',
 'LEC',
 'CBLOL',
 'LFL']
```
Just in case to confirm that other features had the same list of leagues that had 100% missgness and the distribution is the same among those same features I wrote this peace of code for killsat15
```
#Lets double check and do the same for goldat15 and compare lists
study_df["killsat15_missing"] = study_df["killsat15"].isna()
league_cols_two = study_df.groupby("league")["killsat15_missing"].mean().sort_values(ascending=False)
list_of_league_na_two = list((league_cols_two == 1.0).index)
list_of_league_na_two == list_of_league_na
```
```
True
```
So my final thought would be: By grouping by league, I found several leagues (ASCI, DCup, LPL, …) where the proportion of missing goldat10 is 1.0, and others where it is 0.0. The same set of leagues has complete missingness for goldat15, which suggests that early-game snapshot stats are simply not logged in those leagues. Therefore, the missingness of goldat10 (and other *at10/*at15 stats) is best described as MAR with respect to league.
#### Missgness For Categorical Columns
```
playerid:  object 2209
playername:  object 16
```
Now knwoing what caused the missgness in the previous numeric features I decided to check why some of the playerid are missing and first to check its relation to leagues in league column:
```
study_df["playerid_missing"] = study_df["playerid"].isna()
league_playerid = study_df.groupby("league")["playerid_missing"].mean().sort_values(ascending=False)
league_playerid_na_list = list((league_playerid == 1.0).index)
print(league_playerid_na_list)
#So there seems to be missgness of some players ids in certain smaller league tournaments
```
```text
['DCup', 'LAS', 'GLLPA', 'PRMP', 'ASCI', 'EBLPA', 'LPLOL', 'VL', 'GL', 'EBL', 'SL (LATAM)', 'CT', 'NLC Aurora Open', 'LJLA', 'ESLOL', 'IC', 'CDF', 'UPL', 'EL', 'TAL', 'PGC', 'LCL', 'LHE', 'HC', 'NEXO', 'LMF', 'DDH', 'VCS', 'LFL2', 'CBLOLA', 'GLL', 'TCL', 'LJL', 'PRM', 'LDL', 'PGN', 'EUM', 'LCO', 'UL', 'PCS', 'NLC', 'LCSA', 'USP', 'LFL', 'MSI', 'LVP SL', 'LPL', 'LLA', 'CBLOL', 'LEC', 'LCS', 'LCKC', 'LCK', 'HM', 'WLDs']
```
Seems like the history repeats itself and shows that indeed a lot of player's ids are simply missing in some of the league tournmanament. The conclusion I thought of is: "playerid is missing for some leagues probably because those competitions don’t use or expose stable numeric player IDs in the stats feed. It’s a logging/metadata issue, not an attendance or performance issue, and that would explain why only some of leagues have it missing. Which indicates to be the case for MAR.".
Moving on to playername before checking on the league column again I wanted to see just how much of playername is available and how much is not:
```
#Now test why some teamname and playername is missing
#Check how rare the missgness is
for col in ["playername"]:
    miss_col = col + "_missing"
    study_df[miss_col] = study_df[col].isna()
    print(f"\n=== {col} ===")
    print(study_df[miss_col].value_counts())
    print("proportion missing:", study_df[miss_col].mean())
```
```text
=== playername ===
playername_missing
False    125474
True         16
Name: count, dtype: int64
proportion missing: 0.0001275001992190613
```
The proportion seemed very, very small, however just in case I decided to see its missgness by league:
```
for col in ["playername", "teamname"]:
    miss_col = col + "_missing"
    print(f"\n=== {col} missingness by position ===")
    print(
        study_df.groupby("position")[miss_col]
        .mean()
        .sort_values(ascending=False)
    )
```
```text
=== playername missingness by position ===
position
mid    5.58e-04
bot    7.97e-05
jng    0.00e+00
sup    0.00e+00
top    0.00e+00
Name: playername_missing, dtype: float64
```
Thoughts: The position scenario tells us that the numbers are so small(less than 0.1%) that There’s no meaningful pattern across roles: playername: a few missing mids/bots, literally zero for others → looks like random missgness. So, with respect to position, missingness looks MCAR (no systematic dependence). Conclusion: To conclude everything, the missgness is extremely rare, and has slightly concentrated in a few small leagues (potential loggin issues or restrign information). playername is missing for just 16 out of ~125k player-games. The missgnes is scatter among couple of columns such as kills and leagues with almost the same kills distributions as non-missing rows. It suggests that the missgness could be at random due to loggin issues or regestring issues on the servers in few tournaments(leagues) and not from any systematic dependence on player performance or in-game role. In other words, missingness in playername is best described as MAR given league, and effectively MCAR with respect to the main variables of interest (kills, position, early-game stats).
## Data Cleaning and Exploratory Data Analysis Part: 2
Now that I assesed the missgness time to finish the EDA, but before that I do quick cleaning for my data frame:
```
#Since we know that for *at10 and *at15 columns are missgness due to some leagues, it is better decision to get rid of them,
#because we have no availiability to it
# Columns we want for modeling (target + early-game features)
target = "kills"

num_feats_10 = ["goldat10", "xpat10", "csat10", "killsat10", "assistsat10", "deathsat10"]
num_feats_15 = ["goldat15", "xpat15", "csat15", "killsat15", "assistsat15", "deathsat15"]

core_cols = [target] + num_feats_10 + num_feats_15

# 1. Start from full study_df and drop rows with missing core numeric values
base = study_df.dropna(subset=core_cols).copy()

# 2. Drop identifier / meta columns we won't use as features
cols_to_drop = ["playerid", "playername", "teamname", "gameid"]

for c in cols_to_drop:
    if c in base.columns:
        base = base.drop(columns=c)

print("Rows in base:", len(base))
print("Columns in base:", base.shape[1])
base
```
```
Rows in base: 106560
Columns in base: 29
```
| date                | league   |   year | split   |   playoffs |   patch | side   | position   |   gamelength |   kills |   goldat10 |   xpat10 |   csat10 |   killsat10 |   assistsat10 |   deathsat10 |   goldat15 |   xpat15 |   csat15 |   killsat15 |   assistsat15 |   deathsat15 |   game_minutes | goldat10_missing   | goldat15_missing   | killsat15_missing   | teamname_missing   | playerid_missing   | playername_missing   |
|:--------------------|:---------|-------:|:--------|-----------:|--------:|:-------|:-----------|-------------:|--------:|-----------:|---------:|---------:|------------:|--------------:|-------------:|-----------:|---------:|---------:|------------:|--------------:|-------------:|---------------:|:-------------------|:-------------------|:--------------------|:-------------------|:-------------------|:---------------------|
| 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | top        |         1713 |       2 |       3228 |     4909 |       89 |           0 |             0 |            0 |       5025 |     7560 |      135 |           0 |             1 |            0 |          28.55 | False              | False              | False               | False              | False              | False                |
| 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | jng        |         1713 |       2 |       3429 |     3484 |       58 |           1 |             2 |            0 |       5366 |     5320 |       89 |           2 |             3 |            2 |          28.55 | False              | False              | False               | False              | False              | False                |
| 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | mid        |         1713 |       2 |       3283 |     4556 |       81 |           0 |             1 |            0 |       5118 |     6942 |      120 |           0 |             3 |            0 |          28.55 | False              | False              | False               | False              | False              | False                |
| 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | bot        |         1713 |       2 |       3600 |     3103 |       78 |           1 |             1 |            0 |       5461 |     4591 |      115 |           2 |             1 |            2 |          28.55 | False              | False              | False               | False              | False              | False                |
| 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | sup        |         1713 |       1 |       2678 |     2161 |       16 |           1 |             1 |            0 |       3836 |     3588 |       28 |           1 |             2 |            2 |          28.55 | False              | False              | False               | False              | False              | False                |
#### Distribution and Histograms
I'll be going over couple of the distributions I consider to be the most important from this graphs however the full list of all histograms you can find the folder called "assets" in the main repo. The first distribution I'll be looking at is the distribution of game_minutes and by looking at it it seems it was correct to assume that most of the games in LoL matches last around 30-35 minutes with very few last longer(around 40-50 do happen but rarely). Thus it seems appropriate to say that early game features at10 and midgame at 15.
The Code that I used to print out the histograms such as game_minutes but also other features:
```
fig_time = px.histogram(
        base,
        x=study_df['game_minutes'],
        nbins=30,
        title=f"Distribution of game_minutes",
        labels={col: "game_minutes", "count": "Number of player-games"},
    )
fig_time.write_html(f"assets/hist_game_minutes.html", include_plotlyjs="cdn")
fig_time.show()
```
<iframe
    src="assets/hist_game_minutes.html"
    width="800"
    height="500"
    frameborder="0">
</iframe>

Next to look is the target column which is kills. 
```
base.groupby("position")["kills"].describe()[["mean", "std", "25%", "50%", "75%"]]
```
|     mean |     std |   25% |   50% |   75% |
|---------:|--------:|------:|------:|------:|
| 4.25788  | 3.24725 |     2 |     4 |     6 |
| 3.09455  | 2.52282 |     1 |     3 |     4 |
| 3.55021  | 2.76407 |     1 |     3 |     5 |
| 0.895927 | 1.15068 |     0 |     1 |     1 |
| 2.79931  | 2.41912 |     1 |     2 |     4 |

<iframe
    src="assets/hist_kills.html"
    width="800"
    height="500"
    frameborder="0">
</iframe>

The kills are very skewed. Most players end the game with only a few kills: tallest bars are between 0-4 kills, and it drops steadily after that. Double digit kills exists but are very rare, and it seems to even have a few outliers with above 15-16 kills. This tells us that typical players have a low kill count, whule small number of players contribute to a long right tail. The variance seems quite high so prediction for outliers can be hard to achieve. And because the target column is continous but skewed regression model is the most appropriate.
##### Position Relation to Kills:
- Kills vary strongly by role: bot lane averages about 4.3 kills, mid about 3.6, jungle 3.1, top 2.8, while supports average <1 kill per game.
- The 75th percentile for bot is 6 kills, compared to only 1 kill for support, showing that supports almost never get many kills even in the upper quartile.
- These stats show us that the role in the game does strongly affect the amount of kills player will be receiving on average thus it would be import to include it in our model and one-hot code it
##### Couple Other Mid Game Stats
<iframe
    src="assets/hist_killsat15.html"
    width="800"
    height="500"
    frameborder="0">
</iframe>
<iframe
    src="assets/hist_goldat15.html"
    width="800"
    height="500"
    frameborder="0">
</iframe>
<iframe
    src="assets/hist_xpat15.html"
    width="800"
    height="500"
    frameborder="0">
</iframe>

- Gold (goldat15)
    - Roughly bell-ish, centered around 3–3.5k at 10 and 5–5.5k at 15.
    - The right tail gets heavier at 15 (some players up near 8–9k+, but rarely).
    - Almost everyone has similar gold early, but as the game progresses, snowball games create huge leads for a small set of players.
- XP (xpat15)
    - Doesn't seem to have a complete bellish form and have around same distribution for between 5-6k and 7-8k.
    - Less extreme tail than gold, which fits the idea that XP is more “controlled” while gold spikes a lot with kills.
- CS (csat15)
    - Clear multi-modal / lumpy shape: one group at very low CS, another much higher.
    - The CS distributions are multi-modal, hinting that roles have systematically different farming patterns. This motivates stratifying by
- Kills (killsat15)
    - Heavily right-skewed, majority of players have almost no kills and a good chunk have around 1-2 kills with much smaller percentage 3-4 kills, and some outliers have 5-6 kills.
##### Scatter Plots
The one I found out to have the biggest correlation are goldat15 and killsat15 with final kills which makes sense because the more gold and kills the player have the higher the chances of him having successfull kill as well as their position correlation to the final kills. 
```
base.groupby("position")["kills"].describe()[["mean", "std", "25%", "50%", "75%"]]
```
|     mean |     std |   25% |   50% |   75% |
|---------:|--------:|------:|------:|------:|
| 4.25788  | 3.24725 |     2 |     4 |     6 |
| 3.09455  | 2.52282 |     1 |     3 |     4 |
| 3.55021  | 2.76407 |     1 |     3 |     5 |
| 0.895927 | 1.15068 |     0 |     1 |     1 |
| 2.79931  | 2.41912 |     1 |     2 |     4 |

This table shows that on average bot lane have stronger differences bots ~ 4.3 kills on average while sup ~ 0.9, which indicates potential relation between player's position and final kills which we will later test in Hypothesis Testing.
The Code that was used for scatter plots:
```
for col in feature_cols:
    fig = px.scatter(
        base,
        x=col,
        y=target,
        color="position",
        opacity=0.4,
        title=f"{target} vs {col}",
        labels={col: col, target: "Kills"},
        trendline="ols",
    )
    fig.write_html(f"assets/kills_vs_{col}.html", include_plotlyjs="cdn")
    fig.show()
```
<iframe
    src="assets/kills_vs_killsat15.html"
    width="800"
    height="500"
    frameborder="0">
</iframe>
<iframe
    src="assets/kills_vs_goldat15.html"
    width="800"
    height="500"
    frameborder="0">
</iframe>
<iframe
    src="assets/kills_vs_xpat15.html"
    width="800"
    height="500"
    frameborder="0">
</iframe>

##### Early gold & XP vs kills
- kills vs goldat10 / goldat15
    - Clear positive relationship: players with more early gold tend to end the game with more kills.
    - Lines by position:
        - Bot and mid have the steepest lines → extra gold converts into more kills for them.
        - Support has the flattest line → supports can get gold without getting many kills themselves.
- kills vs xpat15
    - Similarity: more early XP → more kills.
     - Mid and bot again tend to sit higher (more kills for the same XP).
    - Top/support sit lower; jungle is in between.
- Interpretation:
    - Early-game resource leads(gold/XP) are moderately positively associated with final kills, and this effect is strongest for damage focused roles (mid, bot).
#### Early kills/assists vs total kills
- kills vs killsat10 / killsat15
    - Almost linear trend: more early kills → many more total kills.
    - It is shown to be one of the strongest relationship visually on graphs
    - Lines by position: bot and mid have the highest lines, support the lowest.
- Interpretation:
    - Early kills seem to have strong relationship with final kill(the more kills you get early the bigger advantage is), while early assists are positively associated but much weaker, especially for the support role due to it being less of the agressive/attack role. 

Now It is time to explain why I mostly show features at15* mark. This is due to the distribution of kills in general and how the game works in general. A lot of actions that happen during the game start happening around mark of 15 minutes, before that players tend to spend a lot of time farming gold and other stuff depending on their position and avoid fighting, thus not a lot infromation can be told to the model during first 10 minutes, thus I shifted my focus more on the mid game features.
## Hypothesis Testing
First I will be doing hypothesis testing on the position's relation to the kills which I one-hot coded in order to be able to use it for the regression model.
##### Position Hypothesis
- I decide to test whether the position affects the kills and here is my null and alternative hypothesis:
    - H₀: Kills are independent of position (all roles have the same mean kills).
    - H₁: At least one role has a different mean kills.
```
def r2_from_design(X, y):
    # least squares solution
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

rng = np.random.default_rng(0)


pos_dummies = pd.get_dummies(base["position"], prefix="pos", drop_first=True)
# columns will be like: pos_bot, pos_jng, pos_mid, pos_sup

X = pos_dummies.to_numpy()
y = base["kills"].to_numpy()

X_design = np.column_stack([np.ones(len(X)), X])

obs_r2 = r2_from_design(X_design, y)
print("Observed R² (kills ~ position dummies):", obs_r2)

# build null distribution by breaking the relationship between kills and position
reps = 2000
null_r2 = np.empty(reps)

for i in range(reps):
    y_perm = rng.permutation(y)  # shuffle kills, keep positions fixed
    null_r2[i] = r2_from_design(X_design, y_perm)

# one-sided p-value: do positions explain more variance than we'd expect by chance?
p_value = np.mean(null_r2 >= obs_r2)
print("Permutation p-value:", p_value)
```
```text
Observed R² (kills ~ position dummies): 0.16632020766961886
Permutation p-value: 0.0
```
Small P value which below 0.01 and since R^2 is ~0.166 inidicates that we reject null hypothesis and cannot say that position has no relationship to the player's kills.
Now I'm going to do Hypothesis testing for my *at10 and *at15 minute early features with general idea for each column X. Clean way to do hypothesis tests for these features is a permutation test on the correlation between each feature and kills.
- H₀: kills and X are independent (true correlation = 0).
- H₁: kills and X are associated (correlation ≠ 0)
```
Helper Function
def perm_corr_pvalue(df, x_col, y_col="kills", reps=2000, seed=0):
    """
    Permutation test for correlation between x_col and y_col.

    Returns (observed_correlation, p_value).
    """
    rng = np.random.default_rng(seed)

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    obs_corr = np.corrcoef(x, y)[0, 1]

    # null distribution: break relationship by shuffling y
    null_corr = np.empty(reps)
    for i in range(reps):
        y_perm = rng.permutation(y)
        null_corr[i] = np.corrcoef(x, y_perm)[0, 1]

    # two-sided p-value
    count = np.sum(np.abs(null_corr) >= abs(obs_corr))
    p_value = (count + 1) / (reps + 1)

    return obs_corr, p_value
```
```
cols_to_test = [
    "goldat10", "xpat10", "csat10", "killsat10", "assistsat10",
    "goldat15", "xpat15", "csat15", "killsat15", "assistsat15",
]

for col in cols_to_test:
    corr, p = perm_corr_pvalue(base, col, y_col="kills", reps=10000, seed=0)
    print(f"{col:10s}  corr = {corr: .4f},  p-value = {p: .4g}")
```
Results:
```text
goldat10    corr =  0.5287,  p-value =  9.999e-05
xpat10      corr =  0.2307,  p-value =  9.999e-05
csat10      corr =  0.3659,  p-value =  9.999e-05
killsat10   corr =  0.4470,  p-value =  9.999e-05
assistsat10  corr =  0.0156,  p-value =  9.999e-05
goldat15    corr =  0.5880,  p-value =  9.999e-05
xpat15      corr =  0.2966,  p-value =  9.999e-05
csat15      corr =  0.3728,  p-value =  9.999e-05
killsat15   corr =  0.5870,  p-value =  9.999e-05
assistsat15  corr =  0.0245,  p-value =  9.999e-05
```
##### Permutation Test Results and One-Hot on the Position
- [ goldat10 ≈ 0.53, goldat15 ≈ 0.59, killsat10 ≈ 0.45, killsat15 ≈ 0.59, csat10 ≈ 0.37, csat15 ≈ 0.37, xpat10 ≈ 0.23, xpat15 ≈ 0.30, assistsat10 ≈ 0.02, assistsat15 ≈ 0.02, All p-values < 1×10⁻⁴ (permutation floor) ]
- Observed R^2 for position was ≈  0.166
- Conclusions:
    - For features such as gold,xp,cs, and early kills we have strong evidence to reject the null hypothesis, as tests show that stats are definitely have strong association with final kills - very hard to obtain these stats by simple sampling.
    - Breakdown of the Realtionships
        - Very Strong r: goldat15, killsat15
        - Strong r: goldat10, killsat10, csat10, csat15, xpat15
        - Moderate r: xpat10
        - Very Weak r: assistat10, assistat15(probably due to being maninly support stat rather than the attacker/aggresive role)
    - The permutation test shows that lane assignment itself explains a chunk of kills variance beyond random noise
    - Across 10_000 permutation replications never was observed correlations as large as the ones in the real data, yielding p_values < 10^-4 for all gold, xp, cs, and early kills features. Therefore reject null hypothesis of no relationship and conclude that early economy/perfomance stats and lane position are all meaningfully associated with how many final kills the players get by the end of the game.

## Framing a Prediction Problem
##### Type: Regression
- Prediction problem:
    - At the 15-minute mark of a League of Legends match, predict how many kills a player will finish the game with.
    - Response variable: kills (total kills at the end of the game).
    - Time of prediction: 15 minutes into the game. All features we use are known by (or before) 15 minutes — e.g., lane position and mid-game stats like gold, XP, CS, kills, and assists at 15 minutes. As well as one-hot coded position columns since I found the existing correlation between the player's position and final kills.
    - Main evaluation metric: RMSE (root mean squared error), because kills are numeric and RMSE directly measures how far our predictions are, on average, from the true kill counts in the same units. We’ll also report R² as a secondary metric to show how much variance in kills our model explains.

## Baseline Model
##### Baseline model (≥ 2 features)
- Conceptually:
    - Features:
        - Lane position (one-hot encoded)
        - Gold at 15 minutes (goldat15)
    - Reasoning:
        - position alone already explained ~16–17% of the variance in kills.
        - goldat15 had the strongest correlation with kills among numeric features and extremely small permutation p-value.
        - Together, they form a simple but meaningful baseline that uses exactly the kind of information a coach or analyst would have at 15 minutes.
    - The Base and Improving models will all be tested using cross-validation technique.
The Code:

```
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
# target
y = base["kills"].to_numpy()
```

Helper Function

```
def cv_linreg_with_position(df, numeric_feats, n_splits=5, random_state=0):
    """
    Cross-validate a linear regression predicting kills from:
      - numeric_feats (list of column names)
      - one-hot position dummies (drop_first=True)
    """
    # numeric part
    X_num = df[numeric_feats]

    # one-hot encode position
    pos_dummies = pd.get_dummies(df["position"], prefix="pos", drop_first=True)

    # combine into one design matrix
    X = pd.concat([X_num, pos_dummies], axis=1).to_numpy()
    y = df["kills"].to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rmses = []
    r2s = []

    print(f"\n=== Feature set: {numeric_feats} + position dummies ===")
    print("Numeric features:", numeric_feats)
    print("Position dummies:", list(pos_dummies.columns))

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        rmses.append(rmse)
        r2s.append(r2)

        print(f"  Fold {fold}: RMSE = {rmse:.3f}, R² = {r2:.3f}")

    print(f"  Mean RMSE = {np.mean(rmses):.3f} (± {np.std(rmses):.3f})")
    print(f"  Mean R²   = {np.mean(r2s):.3f} (± {np.std(r2s):.3f})")

    return np.mean(rmses), np.mean(r2s)
```

```
feature_sets = [
    (["goldat15"], "baseline_gold15+pos"), # base line or base model, next are imporvements that could help the model
    (["goldat15", "xpat15"], "+xpat15"),
    (["goldat15", "xpat15", "csat15"], "+csat15"),
    (["goldat15", "xpat15", "csat15", "killsat15"], "+kills15"),
    (["goldat15", "xpat15", "csat15", "killsat15", "assistsat15"], "+assists15"),
]

for feats, label in feature_sets:
    print(f"\n>>> Feature set label: {label}")
    cv_linreg_with_position(base, feats, n_splits=5, random_state=0)
```

And so the base line model showed these results:

```text
=== Feature set: ['goldat15'] + position dummies ===
Numeric features: ['goldat15']
Position dummies: ['pos_jng', 'pos_mid', 'pos_sup', 'pos_top']
  Fold 1: RMSE = 2.184, R² = 0.372
  Fold 2: RMSE = 2.182, R² = 0.369
  Fold 3: RMSE = 2.200, R² = 0.376
  Fold 4: RMSE = 2.209, R² = 0.369
  Fold 5: RMSE = 2.191, R² = 0.355
  Mean RMSE = 2.193 (± 0.010)
  Mean R²   = 0.368 (± 0.007)
```

And as adding more features the RMSE and R² decreased a little bit but not much, except when adding killsat15, which descreased both RMSE (went down from 2.193 to 2.026) and increased R² from 0.368 to 0.460.

```
=== Feature set: ['goldat15', 'xpat15', 'csat15', 'killsat15', 'assistsat15'] + position dummies ===
Numeric features: ['goldat15', 'xpat15', 'csat15', 'killsat15', 'assistsat15']
Position dummies: ['pos_jng', 'pos_mid', 'pos_sup', 'pos_top']
  Fold 1: RMSE = 2.020, R² = 0.462
  Fold 2: RMSE = 2.018, R² = 0.460
  Fold 3: RMSE = 2.028, R² = 0.469
  Fold 4: RMSE = 2.038, R² = 0.463
  Fold 5: RMSE = 2.025, R² = 0.449
  Mean RMSE = 2.026 (± 0.007)
  Mean R²   = 0.460 (± 0.007)
```

## Final Model
So a very reasonable final model is:
- Predict kills using:
    - goldat15, xpat15, csat15, killsat15, assistsat15
    + one-hot encoded position dummies (pos_jng, pos_mid, pos_sup, pos_top).
Code skeleton for fitting that model on the full base DataFrame:

```
final_num_feats = ["goldat15", "xpat15", "csat15", "killsat15", "assistsat15"]

# one-hot encode position
pos_dummies = pd.get_dummies(base["position"], prefix="pos", drop_first=True)

X_num = base[final_num_feats]
X_final = pd.concat([X_num, pos_dummies], axis=1)
y = base["kills"]

final_model = LinearRegression()
final_model.fit(X_final, y)

# in-sample performance
y_pred = final_model.predict(X_final)
rmse_full = mean_squared_error(y, y_pred, squared=False)
r2_full = r2_score(y, y_pred)

print("Final model (full data) RMSE:", rmse_full)
print("Final model (full data) R^2 :", r2_full)

# store residuals/errors for fairness
base["y_pred_final"] = y_pred
base["resid_final"] = base["kills"] - base["y_pred_final"]
base["abs_err_final"] = base["resid_final"].abs()
```

```
Final model (full data) RMSE: 2.0258948889737414
Final model (full data) R^2 : 0.4607200638178277
```

## Fairness Analysis

