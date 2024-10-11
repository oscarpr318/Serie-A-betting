Betting algorithm for the Italian Serie A league - the top flight of football in Italy. 

**How it works**

It performs numerical analysis on historical leagues, then uses a Poisson distribution to obtain probabilities for each outcome, and compares these to the odds implied by the bookmaker (after their margin has been removed). Now we can find opportunities for +EV bets.

**Example**

Lazio v Sassuolo (26/05/24)

**Bookmakers closing odds (odds right before game starts):**

**Home win -> 1.36\
Draw -> 5.25\
Away win -> 7.50\
Over 2.5 goals -> 1.50\
Under 2.5 goals -> 2.63**

Note - decimal odds are the reciprocal of probability, and (decimal odds - 1) gives the profit in units payed to us if we win.

----------------------------------------------------------------

To remove the bookmakers margin, we first calculate the margins:

**M = 1/HomeWin + 1/Draw + 1/Awaywin - 1 = 0.059\
M = 1/O25 + 1/U25 - 1 = 0.047**

Then, for the match outcome market, we can use the formula

**FairOdds = (3 x BookieOdds) / (3 - M x BookieOdds)**

Or for the over/under market,

**FairOdds = (2 x BookieOdds) / (2 - M x BookieOdds)**

So the fair odds for this match are:

**Home win -> 1.40, implied prob = 0.71\
Draw -> 5.86, implied prob = 0.17\
Away win -> 8.80, implied prob = 0.11\
Over 2.5 goals -> 1.55, implied prob = 0.65\
Under 2.5 goals -> 2.80, implied prob = 0.36**

----------------------------------------------------------------

Now here are our estimated probabilities:

**Home win -> 0.61
Draw -> 0.21\
Away win -> 0.18\
Over 2.5 goals -> 0.59\
Under 2.5 goals -> 0.41**

We are only interested in draws and over/under goals since these bets return the greatest profit when tested against the previous season. Here our draw prob of 21% is larger than the bookies prob of 17%, and so is our prob for under 2.5 goals (41% > 36%).
These are +EV bets since the bookmaker is paying more money than they should be, so we should make money betting at these odds and probabilities in the long run. Indeed both bets won (the result was 1 : 1). 
For the amount to stake we simply use the Kelly Criterion (https://en.wikipedia.org/wiki/Kelly_criterion).
