# Learning notes: the fourth-down bot

These notes are for explaining the project out loud. Each section says what the idea is, why it is in this repo, and how a good answer sounds if someone asks. The numbers are from the published run (`results/summary.json` and the queries behind the README). If you refit, quote the new file, not this page.

If you have already built a shot-quality model with logistic regression and a gradient-boosted tree, the new pieces here are the decision, not another classifier. You still predict a probability. You then price three different futures, pick the best one, and grade the human who picked something else. The tools that are new relative to a shot model are monotone XGBoost, an isotonic distance curve, binomial GLMs, an empirical punt distribution, a clock rule, and a season holdout that never shuffles rows inside a year.

## Win probability

Win probability is the chance the team with the ball wins the game, from this snap, before you know the play. It is a number between 0 and 1. A tied home team at midfield to start the game is a little above one half, because of home field. The same team, down 7, with a minute left, at their own 10, is far below one half.

In this project the label is simple. If the offense's team won the game, the label is 1. If they lost, it is 0. Ties are dropped, because a tie is neither. Overtime plays are dropped, because the overtime rules are a different game and there are not many of them. The features are things you could write on a card before the snap: score, clock, yard line, down, distance, timeouts, who is home, and the closing point spread.

The model is a gradient-boosted tree (XGBoost) with monotone constraints. A constraint is a promise. A larger lead is not allowed to lower predicted win probability. Being farther from the opponent's goal is not allowed to raise it. The spread, home field, and offensive timeouts are constrained the same way. The tree can still be wrong. It cannot be wrong in a direction you have already ruled out.

Two extra columns do a job the tree is bad at when the data are thin. `score_time` is the lead divided by a slow function of time remaining, so a 7-point lead in the first quarter and a 7-point lead with a minute left are different inputs even before the tree starts splitting. `field_time` does the same for field position. Being on the opponent's 40 with two minutes left is a different play from being there with a full half. The first version of this model, without `field_time` and with shallower trees, barely moved when you walked the offense down the field in the first quarter. Score ate the splits. The published model does move: early, a tied home team on 1st-and-10 is about 0.61 at the opponent's 10 and about 0.50 at its own 10. Late in the fourth (three minutes left) the same walk is about 0.72 against 0.49.

**Interview question.** "What is win probability, in one sentence, and what did you use as the label?"

**A good answer.** "It is the offense's chance of winning the game from this snap. The label is whether that team won, from the final score. I dropped ties and overtime. I did not use nflfastR's win probability as an input. I compared myself to it afterwards."

**Follow-up.** "Why is your log loss better than nflfastR `wp` and worse than `vegas_wp`?"

**A good answer.** "Because I use the point spread and plain `wp` does not. `vegas_wp` uses the spread too. On 303,518 held-out scrimmage plays my log loss is 0.452, `vegas_wp` is 0.444, and plain `wp` is 0.481. Correlation with `vegas_wp` is 0.988. I am close to the spread-aware public model and I do not beat it. 2022 is the only season I do, and that year every model is bad (mine 0.510, vegas 0.516, plain wp 0.556)."

## Expected points versus win probability

Expected points answer a different question: how many points is the next score worth, from this field position and down? A 4th-and-1 at midfield might be worth a fraction of a point more if you go for it than if you punt. That is an expected-points argument. It does not know the score or the clock. Going for it up 21 in the first quarter and going for it down 3 with a minute left are the same expected-points problem and completely different win-probability problems.

This project optimizes win probability. A field goal down 3 with two minutes left ties the game. A field goal up 3 adds insurance. A field goal down 2 takes the lead and gives the ball back. Expected points would treat those kicks as the same three points.

The cost of that choice shows up in the opening-kickoff chart. Early in a tied game, three points followed by a kickoff barely moves this model's win probability (about 51.5% for the team that just scored). Keeping the drive, with a chance at seven, is worth a bit more. So the chart says go across the opponent's half, and the field-goal color never appears. With five minutes left in a tied game, the field-goal band is there, because three points now change the chance of winning. That is the win-probability model doing what it was asked to do. It is also a place to be humble: `vegas_wp` thinks an early three-point lead is worth a few points more than this model does, and that gap is enough to color some of those cells gold. Say that out loud before someone else does. The details are in the README limitations.

**Interview question.** "Why not just maximize expected points on 4th down?"

**A good answer.** "Expected points ignore the score and the clock. The decision I want is the one that wins the game. Down 3 with two minutes left, my chart drops the punt entirely: 723 of 990 cells say go and 267 say kick, and a kick only ties. On the opening kickoff the same machinery says a field goal is worth about the same as going for it, within a few points, because three points with 58 minutes left do not move win probability much. I would rather show that chart and explain the early-score slope than pretend every situation has the same field-goal region."

## The conversion model

You cannot price "go for it" without a chance the play works. The conversion model is the chance a 4th-down pass or run picks up the first down or scores, as a function of yards to go and field position.

It is a binomial GLM. Each play is a coin flip with a probability that depends on the covariates, and the link is logistic, so the output stays between 0 and 1. There are two of them: P(any success) and P(touchdown). Yards to go enter as a line and a square, because 4th-and-1 to 4th-and-2 is a bigger drop than 4th-and-9 to 4th-and-10. Goal-to-go has its own coefficient, because the defense can sell out at the goal line. A year term lets the league-wide rate drift.

The sample is 4th down only. An early bug counted every pass and run in the game as a "go," which made conversion look like an ordinary down. Ordinary downs convert more often than 4th downs, because the defense does not know it is a money down in the same way, and because 3rd-and-10 is a different call from 4th-and-10. The published fit is 7,944 fourth-down attempts, 52.7% successful, 9.0% touchdowns.

At midfield in the 2025 fit: 68.6% on 4th-and-1, 48.4% on 4th-and-5, 29.1% on 4th-and-10. At the own 40, 4th-and-1 is 69.6%.

What happens when it fails also matters. If you turn it over on downs, the opponent takes the ball near the line of scrimmage. If you throw an interception, they might take it somewhere else. If they return it for a touchdown, the score changes. Those are not the same next state. Among failures here, 90.5% are turnovers on downs, 8.2% are giveaways, 0.9% are defensive touchdowns, and 0.4% are safeties. The decision engine mixes those.

Touchdowns are not always 7. The historical mix after an offensive touchdown is 9.1% for 6 points, 86.7% for 7, and 4.2% for 8. The engine rolls that mix into the kickoff that follows.

**Interview question.** "Why a GLM instead of another tree? And why league average?"

**A good answer.** "Yards to go and field position are smooth. A GLM with a square term is easy to plot and easy to defend, and I have a few thousand attempts, not a few hundred thousand. A tree would fit noise at rare spots, like 4th-and-9 from the 3. I kept it league-wide so the chart is 'what does an average NFL offense do,' not 'what do the 2024 Bills do.' If I were advising one team I would add their offense and their kicker. I would say that in the room."

## Field goals and punts

Field-goal distance is the yard line plus 17: the end zone is 10 yards deep and the hold is 7 yards behind the line. Make probability falls as distance rises. Isotonic regression is a curve that is forced to fall (or at least not rise) as the kick gets longer, without assuming a straight line on the logit scale. The published curve: 96.3% at 30 yards, 87.7% at 40, 72.1% at 50, 61.3% at 55, 43.8% at 60. Overall, 12,629 kicks, 84.7% made.

A second GLM sits on top of that curve and asks whether wind, a dome, cold, or Denver move the probability after distance is accounted for. Wind is slightly negative and not distinguishable from noise (p = 0.42). A dome is zero. Denver is the wrong sign and noisy. Cold is positive and the p-value is 0.001, which is the opposite of the barstool theory. Say so. Coaches choose which kicks to attempt in the cold, so the attempts that remain are not a random sample. On a check of 400 cold field goals in the app model, turning the cold flag off flipped 3.5% of the recommendations. The coefficient is in the fit. It is not a finding I would put on a slide as "kick more when it is cold."

A miss is not "the opponent gets the ball at the spot of the kick" in a vague sense. The rule used here matches the spotting convention: 8 yards behind the line of scrimmage, and at the offense's 20 if that would put the ball inside the 20. Kicks past 68 yards are turned off. Otherwise a 70-yard miss, spotted at the 20, can look like a free punt that also has a tiny chance of three points, and the model will recommend absurd kicks.

Punts are a distribution, not a mean. From a given yard line the next snap is sometimes a touchback, sometimes a coffin corner, sometimes a return, rarely a block, rarely a return touchdown. Averaging those into one "net 40 yards" hides the pile of touchbacks. The model stores quantiles of what actually happened. There are 27,056 punts in the fit. Median time from snap to the next scrimmage play is 9 seconds. That clock detail matters late: a failed 4th down with a few seconds left does not hand the opponent a full drive.

**Interview question.** "Walk me through a missed field goal in your model."

**A good answer.** "Distance is yardline plus 17. Make probability comes from the isotonic curve, then a small weather adjustment. If it is good, we add 3 and the other team receives a kickoff, with the touchback spot for that season: the 20 through 2015, the 25 from 2016 to 2023, the 30 from 2024. If it is missed or blocked, they get the ball 8 yards behind our line, unless that is inside our 20, in which case it is the 20. If the distance is over 68 I set the whole option to zero so it cannot win the comparison. Then I ask the win-probability model what that next state is worth."

## Calibration and log loss

A model that says 70% should win about 70% of the time. That is calibration. The calibration plot bins the predictions and plots the observed win rate against the average prediction. The diagonal is perfect. Mean absolute error across ten equal-count bins, play-weighted across the eight test seasons, is 0.025. So a typical bin is off by about two and a half percentage points.

Log loss punishes confident mistakes more than Brier score does. If you say 99% and the team loses, log loss explodes. Brier score is the average squared error, and it is gentler. I report both. Held-out Brier is 0.150 here, 0.147 for `vegas_wp`, 0.162 for plain `wp`.

Platt scaling is a one-variable logistic regression on the logit of the raw probability, fit on a season the trees did not train on. It is there to fix "the model says 80% but those teams win 70%." In this run it did not improve held-out log loss: 0.450 before, 0.452 after, play-weighted. Some years the raw model was already close (2020 calibration slope about 0.98). The app bundle's slope on 2025 is 0.89, a small shrink toward one half. I left the scaling in because the calibration plot is part of the argument, and I say plainly that it did not buy log loss. Hiding that would be worse than the 0.002.

**Interview question.** "Your calibration slope is below 1. What does that mean, and did you overfit the fix?"

**A good answer.** "Below 1 means the raw probabilities were a bit too extreme, so the fix pulls them toward 50%. I fit that line on the previous season only. I did not fit it on the test season. And I checked whether it helped. On the held-out seasons it did not help log loss. The plot still sits near the diagonal, with a bin error of about 0.025. I would rather show the before-and-after than claim the calibration step was a big win."

## Season-based validation

Random row splits leak. A 2024 play and another 2024 play from the same game share the weather, the coaches, the quarterbacks, and the season's rules. If both can land in the training set, the test score is optimistic. The split here is by season.

For test season T:

- trees train on seasons through T−2
- T−1 is the early-stopping and Platt season
- T is scored once and then left alone
- field goals, conversions, and punts for that grade use only seasons before T

The first test season is 2018, so 2017 exists as a calibration year. Coach grades from 2018 through 2025 are all out of season.

The app is a different object. Its trees train through 2024 and scale on 2025, and its component models see every season. That is the right bundle for "what should we think on Sunday," and the wrong bundle for "how did this coach do in 2024." The README uses the strict grades for coaches and labels the single-spot table as the app bundle.

**Interview question.** "Why not k-fold?"

**A good answer.** "A random fold lets week 3 and week 10 of the same season train each other. The kicking environment, the extra-point rule, and the kickoff spot all move by year. I want the test to be a year the model has not lived through. That is what a front office would actually face next September. It also means I have eight test scores instead of one, and I can see that 2022 was hard for everyone."

## Decision analysis

For one 4th down the engine builds three numbers.

**Go.** Mix P(touchdown), P(first down), P(turnover on downs), P(giveaway), P(defensive touchdown), and P(safety). Each branch has a next state: we kick off after a score, or we snap again on 1st down, or they snap from a shifted yard line, or they free-kick after a safety. Win probability of the branch comes from the win-probability model. The value of going is the weighted sum.

**Punt.** Same idea, with the quantile distribution for that yard line. Most of the weight is "they have the ball at the next yard line, score unchanged." A little weight is a score either way.

**Kick.** P(make) times the state "we are up 3, they receive the kickoff," plus P(miss) times the spotted miss. Unavailable past 68 yards.

The recommendation is the largest of the three. The edge is the gap to the second. Win probability left on the table is the gap between the best and the choice the coach actually made. If they matched, it is zero. It is never negative.

A clear call means the edge is at least 0.02, two percentage points. Below that, calling a coach "wrong" is louder than the model. The leaderboard also drops the last 30 seconds of a half, 4th-and-more-than-20, and plays where the pre-snap win probability is under 2% or over 98%. Garbage time would dominate a coach's average if you let it, and it is not where the decision is interesting.

The headline from the grades: when the model says punt, coaches punt 97.3% of the time and leave 0.03 percentage points per play. When it says field goal, they kick 82.6% of the time and leave 0.30. When it says go, they go 27.1% of the time and leave 1.44. The whole "coaches are too conservative" claim is that third row. On the clear go calls only, the go rate is 39.7%, up from 31.3% in 2018 to 54.2% in 2025.

The 4th-and-1 check is there so the trend is not an artifact of the model getting more aggressive by itself. Between the 40s, the go rate on 4th-and-1 went from 39.8% in 2018 to 70.0% in 2025. No win-probability model is required to count that.

**Interview question.** "Belichick grades out as leaving the most win probability. Is your model biased against conservative coaches?"

**A good answer.** "The conversion rate is the league rate, not 'what aggressive coaches do.' If going for it really has a higher win probability, a coach who punts leaves that gap, whoever they are. Belichick's sample is 2018–2023, 556 plays, 1.20 percentage points per play, standard error 0.08. He went on 22.8% of clear go calls. The league in those same years went on 36.7%. So he was conservative relative to his own era, not just relative to 2025. Dan Campbell, 470 plays, is at 0.64 per play and went on 62% of clear go calls. I would also say the standard error pretends plays are independent, and I would not fire anyone over four tenths of a percentage point."

**Follow-up.** "Why not add the win probability up and call it wins?"

**A good answer.** "Because the plays are in the same games. Twenty fourth downs in a blowout are not twenty independent experiments. I report the per-play mean. 0.84 percentage points per play on 24,749 plays is the summary. The sum is a temptation, and it is not a win total."

## A few more questions worth rehearsing

**"How do you know the spread sign is right?"**

"nflverse `spread_line` is the expected home margin. I checked that teams with a positive spread win by a positive margin on average. In the model I flip it for the away team so positive always means the offense is favored. Indianapolis in the Steichen example was a 9.5-point underdog, which is why kicking to go up 1, and then giving Houston the ball, is only a 19.6% win probability in the grade. `vegas_wp` on that snap was 0.37. Plain `wp`, which ignores the spread, was 0.56. Those three numbers are the whole reason to carry the spread."

**"What would you do with another month?"**

"I would put a simple score-and-time baseline under the tree so early three-point leads cannot shrink all the way toward a tie. That is the opening-chart issue. I would fit kicker and offense as partial pooling, and I would show the chart both ways. I would add overtime as its own small model once I trusted the regulation one. I would not add more weather terms. The ones I have are either noise or the wrong sign."

**"What broke while you were building it?"**

"Three things I would volunteer. First, the conversion model was briefly fit on every pass and run, so 4th-down conversion looked too easy. The fix was to restrict the sample to 4th down, and the published rate is 52.7% on 7,944 plays. Second, a very long field goal was being treated as a real option, and the miss spot at the 20 made it look like a punt with a lottery ticket. Anything past 68 yards is now unavailable. Third, a shallow win-probability tree ignored field position early in the game. `field_time` and a deeper tree fixed the shape. I also had a Python bug where the punt model overwrote the punt method, which would have crashed the pipeline. The tests cover the clock, the miss spot, the spread sign, and a backed-up punt versus a goal-line go."

**"Is 4th-and-1 from your own 40 a go?"**

"In the app bundle, yes, by 2.0 percentage points: 0.517 go, 0.497 punt. The conversion chance there is about 70%, and a punt from that area has been leaving the opponent around their own 18. Two points is exactly my cutoff for a clear call. I would describe it as a lean, not a commandment. 4th-and-10 from the own 10 is a punt, 0.458 against 0.432. 4th-and-1 from the 2 is a go by 9 points, 0.606 against a field goal at 0.515."

## How the code is laid out, if they ask you to open a file

| File | What you say |
| --- | --- |
| `src/fourth_down/data.py` | Loads the slim parquet, attaches the next scrimmage play, builds the modeling columns. |
| `src/fourth_down/features.py` | Spread sign, home team, cold, altitude, and the win-probability matrix. No leaked future columns. |
| `src/fourth_down/wp.py` | XGBoost plus Platt scaling. Early stopping uses the calibration season. |
| `src/fourth_down/conversion.py` | The two GLMs and the historical mix of what a success or a failure looks like. |
| `src/fourth_down/field_goal.py` | Isotonic distance curve, weather offset, miss residual. |
| `src/fourth_down/punt.py` | Quantile spots by yard line. |
| `src/fourth_down/decide.py` | The weighted next-state win probability. This is the file to be ready to walk. |
| `src/fourth_down/pipeline.py` | Season folds, the app bundle, tables, figures. |
| `sql/` | The coach and team queries. Worth opening if they want to see that you can write the aggregation yourself. |
| `app/streamlit_app.py` | One situation. Three bars. The recommendation is the max. |

Run the tests with `pytest` before you demo. They do not need the full pipeline. The pipeline is what produces `results/` and `figures/`.
