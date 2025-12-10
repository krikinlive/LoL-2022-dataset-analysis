# Leauge Of Legends 2022 Data Set Analysis - How Can Early Game Features and Positions of The Players Help Us Predicit Their Final Kills
by: Roman Lysenko; email: rlysenko@ucsd.edu
## Introduction
In this project of mine I went over dataset that is publically available by oracleselixir(oracleselixir.com) one league of legends games. The data set provides vast information on player's kills, assists, the position in each uniquly id games/tournaments. The team they are in and so on. However my main goal was to understand how much is it possible, using linear regression model, to predict player's final number of kills using only early-mid game features/information and their in-game position. I understand that league of legends is heavily team base game and so information of team's perfomance is alos can be quite influental towards player's perfomances, however I wanted to see how much is it possible to predicit using only individual features.
## Data Cleaning and Exploratory Data Analysis
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
study_df = lol_df.loc[lol_df["position"] != "team", player_cols] #Get rid of column teams so we do not deal with team rows
study_df
```
'| gameid                | date                | league   |   year | split   |   playoffs |   patch | side   | position   | teamname          | playerid                                  | playername   |   gamelength |   kills |   goldat10 |   xpat10 |   csat10 |   killsat10 |   assistsat10 |   deathsat10 |   goldat15 |   xpat15 |   csat15 |   killsat15 |   assistsat15 |   deathsat15 |\n|:----------------------|:--------------------|:---------|-------:|:--------|-----------:|--------:|:-------|:-----------|:------------------|:------------------------------------------|:-------------|-------------:|--------:|-----------:|---------:|---------:|------------:|--------------:|-------------:|-----------:|---------:|---------:|------------:|--------------:|-------------:|\n| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | top        | BRION Challengers | oe:player:38e0af7278d6769d0c81d7c4b47ac1e | Soboro       |         1713 |       2 |       3228 |     4909 |       89 |           0 |             0 |            0 |       5025 |     7560 |      135 |           0 |             1 |            0 |\n| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | jng        | BRION Challengers | oe:player:637ed20b1e41be1c51bd1a4cb211357 | Raptor       |         1713 |       2 |       3429 |     3484 |       58 |           1 |             2 |            0 |       5366 |     5320 |       89 |           2 |             3 |            2 |\n| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | mid        | BRION Challengers | oe:player:d1ae0e2f9f3ac1e0e0cdcb86504ca77 | Feisty       |         1713 |       2 |       3283 |     4556 |       81 |           0 |             1 |            0 |       5118 |     6942 |      120 |           0 |             3 |            0 |\n| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | bot        | BRION Challengers | oe:player:998b3e49b01ecc41eacc392477a98cf | Gamin        |         1713 |       2 |       3600 |     3103 |       78 |           1 |             1 |            0 |       5461 |     4591 |      115 |           2 |             1 |            2 |\n| ESPORTSTMNT01_2690210 | 2022-01-10 07:44:08 | LCKC     |   2022 | Spring  |          0 |   12.01 | Blue   | sup        | BRION Challengers | oe:player:e9741b3a238723ea6380ef2113fae63 | Loopy        |         1713 |       1 |       2678 |     2161 |       16 |           1 |             1 |            0 |       3836 |     3588 |       28 |           1 |             2 |            2 |'
## Assessment of Missingness

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis
