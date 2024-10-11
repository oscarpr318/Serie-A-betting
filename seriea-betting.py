import pandas as pd
import warnings
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import matplotlib

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

data = pd.DataFrame()

final_year = 23
for year in range(22,final_year):
        data = pd.concat((data, pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{year}{year+1}/I1.csv")))

test_entire_season = True # If true this will backtest the model against the season <final_year>/<final_year + 1> i.e. 23/24
                          # Otherwise bets can be found for individual matches in the future and final_year should be set to 25 (so the model uses data up to all matches played in the 24/25 season)

def _round(x):
        return float('{:.3f}'.format(x))

# Function for calculating probabilities

def calc_odds(data, homeTeam, awayTeam):
             
        homeTeamMatches = data[((data["HomeTeam"] == homeTeam) | (data["AwayTeam"] == homeTeam))]
        
        indices_to_drop = []
        for y in range(len(homeTeamMatches["HomeTeam"])):
            if abs(homeTeamMatches["FTHG"].iloc[y] - homeTeamMatches["FTAG"].iloc[y]) > 3:
                indices_to_drop.append(y)

        homeTeamMatches.drop(index=homeTeamMatches.index[indices_to_drop],inplace=True)
        
        
        
        awayTeamMatches = data[((data["HomeTeam"] == awayTeam) | (data["AwayTeam"] == awayTeam))]
        
        indices_to_drop = []
        for y in range(len(awayTeamMatches["HomeTeam"])):
            if abs(awayTeamMatches["FTHG"].iloc[y] - awayTeamMatches["FTAG"].iloc[y]) > 3:
                indices_to_drop.append(y)
        
        awayTeamMatches.drop(index=awayTeamMatches.index[indices_to_drop],inplace=True)

        ##################
        
        h_xG = [homeTeamMatches["FTHG"].iloc[x] for x in range(len(homeTeamMatches["FTHG"])) if homeTeamMatches["HomeTeam"].iloc[x] == homeTeam] + [homeTeamMatches["FTAG"].iloc[x] for x in range(len(homeTeamMatches["FTAG"])) if homeTeamMatches["AwayTeam"].iloc[x] == homeTeam]
        if len(h_xG) == 0:
            return -1,0,0,0,0
        
        h_xG = sum(h_xG) / len(h_xG)
        
        a_xG = [awayTeamMatches["FTHG"].iloc[x] for x in range(len(awayTeamMatches["FTHG"])) if awayTeamMatches["HomeTeam"].iloc[x] == awayTeam] + [awayTeamMatches["FTAG"].iloc[x] for x in range(len(awayTeamMatches["FTAG"])) if awayTeamMatches["AwayTeam"].iloc[x] == awayTeam]
        if len(a_xG) == 0:
            return -1,0,0,0,0
        a_xG = sum(a_xG) / len(a_xG)
        
        ##################
        
        h_xGC = [homeTeamMatches["FTAG"].iloc[x] for x in range(len(homeTeamMatches["FTAG"])) if homeTeamMatches["HomeTeam"].iloc[x] == homeTeam] + [homeTeamMatches["FTHG"].iloc[x] for x in range(len(homeTeamMatches["FTHG"])) if homeTeamMatches["AwayTeam"].iloc[x] == homeTeam]
        if len(h_xGC) == 0:
            return -1,0,0,0,0
        h_xGC = sum(h_xGC) / len(h_xGC)
        
        a_xGC = [awayTeamMatches["FTAG"].iloc[x] for x in range(len(awayTeamMatches["FTAG"])) if awayTeamMatches["HomeTeam"].iloc[x] == awayTeam] + [awayTeamMatches["FTHG"].iloc[x] for x in range(len(awayTeamMatches["FTHG"])) if awayTeamMatches["AwayTeam"].iloc[x] == awayTeam]
        if len(a_xGC) == 0:
            return -1,0,0,0,0
        a_xGC = sum(a_xGC) / len(a_xGC)
        
        ##################
        
        h_avgHomeGoals = [homeTeamMatches["FTHG"].iloc[x] for x in range(len(homeTeamMatches["FTHG"])) if homeTeamMatches["HomeTeam"].iloc[x] == homeTeam]
        if len(h_avgHomeGoals) == 0:
            return -1,0,0,0,0
        h_avgHomeGoals = sum(h_avgHomeGoals) / len(h_avgHomeGoals)
        
        h_avgAwayGoals = [homeTeamMatches["FTAG"].iloc[x] for x in range(len(homeTeamMatches["FTAG"])) if homeTeamMatches["AwayTeam"].iloc[x] == homeTeam]
        if len(h_avgAwayGoals) == 0:
            return -1,0,0,0,0
        h_avgAwayGoals = sum(h_avgAwayGoals) / len(h_avgAwayGoals)

        ##################

        h_avgHomeGoalsConceded = [homeTeamMatches["FTAG"].iloc[x] for x in range(len(homeTeamMatches["FTAG"])) if homeTeamMatches["HomeTeam"].iloc[x] == homeTeam]
        if len(h_avgHomeGoalsConceded) == 0:
            return -1,0,0,0,0
        h_avgHomeGoalsConceded = sum(h_avgHomeGoalsConceded) / len(h_avgHomeGoalsConceded)
        
        h_avgAwayGoalsConceded = [homeTeamMatches["FTHG"].iloc[x] for x in range(len(homeTeamMatches["FTHG"])) if homeTeamMatches["AwayTeam"].iloc[x] == homeTeam]
        if len(h_avgAwayGoalsConceded) == 0:
            return -1,0,0,0,0
        h_avgAwayGoalsConceded = sum(h_avgAwayGoalsConceded) / len(h_avgAwayGoalsConceded)
        
        ##################

        a_avgHomeGoals = [awayTeamMatches["FTHG"].iloc[x] for x in range(len(awayTeamMatches["FTHG"])) if awayTeamMatches["HomeTeam"].iloc[x] == awayTeam]
        if len(a_avgHomeGoals) == 0:
            return -1,0,0,0,0
        a_avgHomeGoals = sum(a_avgHomeGoals) / len(a_avgHomeGoals)
        
        a_avgAwayGoals = [awayTeamMatches["FTAG"].iloc[x] for x in range(len(awayTeamMatches["FTAG"])) if awayTeamMatches["AwayTeam"].iloc[x] == awayTeam]
        if len(a_avgAwayGoals) == 0:
            return -1,0,0,0,0
        a_avgAwayGoals = sum(a_avgAwayGoals) / len(a_avgAwayGoals)

        ##################

        a_avgHomeGoalsConceded = [awayTeamMatches["FTAG"].iloc[x] for x in range(len(awayTeamMatches["FTAG"])) if awayTeamMatches["HomeTeam"].iloc[x] == awayTeam]
        if len(a_avgHomeGoalsConceded) == 0:
            return -1,0,0,0,0
        a_avgHomeGoalsConceded = sum(a_avgHomeGoalsConceded) / len(a_avgHomeGoalsConceded)
        
        a_avgAwayGoalsConceded = [awayTeamMatches["FTHG"].iloc[x] for x in range(len(awayTeamMatches["FTHG"])) if awayTeamMatches["AwayTeam"].iloc[x] == awayTeam]
        if len(a_avgAwayGoalsConceded) == 0:
            return -1,0,0,0,0
        a_avgAwayGoalsConceded = sum(a_avgAwayGoalsConceded) / len(a_avgAwayGoalsConceded)
        
        lmb = h_xG * a_xGC 
        mu = a_xG * h_xGC

        home_probs = poisson.pmf(list(range(10)), lmb)
        away_probs = poisson.pmf(list(range(10)), mu)

        h_predGoals = home_probs.argmax(axis=0)
        a_predGoals = away_probs.argmax(axis=0)

        probability_matrix = np.outer(home_probs, away_probs)

        u25_odds = probability_matrix[0][0] + probability_matrix[0][1] + probability_matrix[1][0] + probability_matrix[1][1] + probability_matrix[0][2] + probability_matrix[2][0] 
        o25_odds = 1 - u25_odds
        
        home_win_odds = np.tril(probability_matrix, -1)
        s = 0
        for y in home_win_odds:
            s += sum(y)
        home_win_odds = s
        
        draw_odds = np.trace(probability_matrix)
        
        away_win_odds = np.triu(probability_matrix, 1) 
        s = 0
        for y in away_win_odds:
            s += sum(y)
        away_win_odds = s

        return home_win_odds, draw_odds, away_win_odds,  o25_odds, u25_odds 

def test_season(data, matches):

    bankrolls = []
    dates = []
    odds = []
    units = []
    net_profits = []
        
    wins = 0
    losses = 0
    bankroll = 50

    exact_result_wins = 0
    exact_result_losses = 0

    for x in range(len(matches["HomeTeam"])):
        
        homeTeam = matches["HomeTeam"].iloc[x]
        awayTeam = matches["AwayTeam"].iloc[x]
        
        homeGoals = matches["FTHG"].iloc[x]
        awayGoals = matches["FTAG"].iloc[x]

        b365HOdds = matches["B365H"].iloc[x]
        b365DOdds = matches["B365D"].iloc[x]
        b365AOdds = matches["B365A"].iloc[x]
        
        b365o25Odds = matches["B365>2.5"].iloc[x]
        b365u25Odds = matches["B365<2.5"].iloc[x]
        
        take = (1/b365o25Odds) + (1/b365u25Odds)  - 1
        
        b365o25Odds_adjusted = 1 / ( 2 * b365o25Odds / (2 - take*b365o25Odds) )
        b365u25Odds_adjusted = 1 / ( 2 * b365u25Odds / (2 - take*b365u25Odds) )
        
        take = (1/b365HOdds) + (1/b365DOdds) + (1/b365AOdds) - 1
        
        b365HOdds_adjusted = 1 / ( 3 * b365HOdds / (3 - take*b365HOdds) )
        b365DOdds_adjusted = 1 / ( 3 * b365DOdds / (3 - take*b365DOdds) )
        b365AOdds_adjusted = 1 / ( 3 * b365AOdds / (3 - take*b365AOdds) )
        
        
        home_win_odds, draw_odds, away_win_odds,  o25_odds, u25_odds  = calc_odds(data, homeTeam,awayTeam)
        
        if home_win_odds == -1:
            continue
    
            
        if o25_odds / b365o25Odds_adjusted > 1.2:
            
            f = o25_odds - (1 - o25_odds)/(b365o25Odds-1)

            unit = _round(bankroll * f)
            
            if unit > 0: 
            
                print(matches["Date"].iloc[x])
                
                print(homeTeam + " v " + awayTeam)
                print(homeGoals, " : ", awayGoals)
                
                print("Bet: O25\nUnit: ", unit, "\nOdds: ", b365o25Odds, "\nProb: ", _round(o25_odds * 100), "%")
            
                if (homeGoals + awayGoals) > 2.5:
                    
                    pnl = unit * (b365o25Odds - 1)
                    
                    wins += 1
                    bankroll += pnl
                                     
                    
                    print("PnL: ", _round(pnl))
                    
                else:
                    losses += 1
                    bankroll -= unit
                    
                    print("PnL: -", unit)
                    
                print()
                    
    
                bankrolls.append(_round(bankroll))
                
                dates.append(matches["Date"].iloc[x])
                
                odds.append(b365o25Odds) 
                
                units.append(unit)
                
                net_profits.append(unit * (b365o25Odds - 1))
                
        
        if u25_odds / b365u25Odds_adjusted > 1:
            
            f = u25_odds - (1 - u25_odds)/(b365u25Odds-1)
            
            unit = _round(bankroll * f)
            
            if unit > 0:
            
                print(matches["Date"].iloc[x])
                
                print(homeTeam + " v " + awayTeam)
                print(homeGoals, " : ", awayGoals)
            
                
                print("Bet: U25\nUnit: ", unit, "\nOdds: ", b365u25Odds, "\nProb: ", _round(u25_odds * 100), "%")
            
                if (homeGoals + awayGoals) < 2.5:
                    
                    pnl = unit * (b365u25Odds - 1)
                    
                    wins += 1
                    bankroll += pnl
                                     
                    
                    print("PnL: ", _round(pnl))
                    
                else:
                    losses += 1
                    bankroll -= unit
                    
                    print("PnL: -", unit)
                    
                print()
                    
    
                bankrolls.append(_round(bankroll))
                
                dates.append(matches["Date"].iloc[x])
                
                odds.append(b365u25Odds) 
                
                units.append(unit)
                
                net_profits.append(unit * (b365u25Odds - 1))
            
        
        if draw_odds / b365DOdds_adjusted > 1.0:
            
            f = draw_odds - (1 - draw_odds)/(b365DOdds-1)

            unit = _round(bankroll * f)
            
            if unit > 0: 
            
                print(matches["Date"].iloc[x])
                
                print(homeTeam + " v " + awayTeam)
                print(homeGoals, " : ", awayGoals)
                
                print("Bet: Draw\nUnit: ", unit, "\nOdds: ", b365DOdds, "\nProb: ", _round(draw_odds * 100), "%")
            
                if homeGoals == awayGoals:
                    
                    pnl = unit * (b365DOdds - 1)
                    
                    wins += 1
                    bankroll += pnl
                                     
                    
                    print("PnL: ", _round(pnl))
                    
                else:
                    losses += 1
                    bankroll -= unit
                    
                    print("PnL: -", unit)
                    
                print()
                    
    
                bankrolls.append(_round(bankroll))
                
                dates.append(matches["Date"].iloc[x])
                
                odds.append(b365DOdds) 
                
                units.append(unit)
                
                net_profits.append(unit * (b365DOdds - 1))
            
             
        
        
        
        match = [matches[column].iloc[x] for column in matches.columns]
        data = pd.concat((data, pd.DataFrame([match],columns=[matches.columns])))

    
    print("Bankroll at ", dates[-1],": ", bankrolls[-1])
    
    print("ROI %: ", _round(sum(net_profits) / sum(units) * 100) ,"\n")

    print("total bets: ", wins+losses,"\nwins: ", wins,"\nlosses: ",losses,"\nwin %:", _round(wins/(wins+losses) * 100),"\n")
        
    print("Avg odds: ",  _round( sum(odds) / len(odds) ))
    
    
                    
    
    
    xpoints = np.array(dates)
    ypoints = np.array(bankrolls)

    plt.plot(xpoints, ypoints)
    plt.ylabel('Bankroll')
    plt.xlabel('Date')
    
    plt.show()
    
    


def is_value_bet(homeTeam,awayTeam, data, b365HOdds, b365DOdds, b365AOdds, b365o25Odds, b365u25Odds):
    
        global bankroll
    
        print(homeTeam + " v " + awayTeam)


        take = (1/b365HOdds) + (1/b365DOdds) + (1/b365AOdds) - 1
         
        b365HOdds_adjusted = 1 / ( 3 * b365HOdds / (3 - take*b365HOdds) )
        b365DOdds_adjusted = 1 / ( 3 * b365DOdds / (3 - take*b365DOdds) )
        b365AOdds_adjusted = 1 / ( 3 * b365AOdds / (3 - take*b365AOdds) )
        
        take = (1/b365o25Odds) + (1/b365u25Odds)  - 1
         
        b365o25Odds_adjusted = 1 / ( 2 * b365o25Odds / (2 - take*b365o25Odds) )
        b365u25Odds_adjusted = 1 / ( 2 * b365u25Odds / (2 - take*b365u25Odds) )
         
        home_win_odds, draw_odds, away_win_odds, o25_odds, u25_odds  = calc_odds(data, homeTeam,awayTeam)

        print("BOOKIES PROBABILITIES: ", _round(b365HOdds_adjusted), _round(b365DOdds_adjusted), _round(b365AOdds_adjusted), _round(b365o25Odds_adjusted), _round(b365u25Odds_adjusted))
        print("MODEL PROBABILITIES: ", _round(home_win_odds), _round(draw_odds), _round(away_win_odds), _round(o25_odds), _round(u25_odds))
        
        
        if o25_odds/b365o25Odds_adjusted > 1:
            print("Value O25")
            
            f = o25_odds - (1 - o25_odds)/(b365o25Odds-1)
            
            print("% of bankroll to wager: ", _round(f * 100))

            
        if u25_odds/b365u25Odds_adjusted > 1:
            print("Value U25")
            
            f = u25_odds - (1 - u25_odds)/(b365u25Odds-1)           
            
            print("% of bankroll to wager: ", _round(f * 100))

        if draw_odds/b365DOdds_adjusted > 1:
            print("Value draw")
            
            f = draw_odds - (1 - draw_odds)/(b365DOdds-1)
            
            print("% of bankroll to wager: ", _round(f * 100))

            
        print("\n")
            
if test_entire_season == True:
    
    matches = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{final_year}{final_year+1}/I1.csv")
    
    test_season(data, matches)
    
else:
    
    # Calculate odds for individual match
    # hts = list of corresponding home teams 
    # ats = list of corresponding away teams
    # bOdds = list of lists of bookmakers odds in order home win, draw, away win, over 2.5, under 2.5
    
    
    hts = []
    ats = []
    bOdds = [[]]
    
    for x in range(len(hts)):
        is_value_bet(hts[x], ats[x], data, bOdds[x][0], bOdds[x][1], bOdds[x][2],bOdds[x][3], bOdds[x][4],)
