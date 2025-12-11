# Leauge Of Legends 2022 Data Set Analysis - How Can Early Game Features and Positions of The Players Help Us Predicit Their Final Kills
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
I'll be going over couple of the distributions I consider to be the most important from this graphs however the full list of all histograms you can find the folder called "assets" in the main repo. 
## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis
