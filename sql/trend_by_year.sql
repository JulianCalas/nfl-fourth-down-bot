-- Has the league become more willing to go for it when the model says to?
-- `go_rate_when_should` is the share of clear "go" recommendations that
-- were actually passes or runs. `go_rate_fourth_and_one` is the same idea
-- with no model at all: fourth and 1 between the opponent's 40 and the
-- offense's 40, so a trend here is not an artifact of the win-probability fit.
SELECT
    season,
    COUNT(*) AS decisions,
    AVG(CASE WHEN went_for_it THEN 1.0 ELSE 0.0 END) AS go_rate_all,
    AVG(CASE WHEN clear_call THEN CASE WHEN followed THEN 1.0 ELSE 0.0 END END) AS follow_rate_clear,
    SUM(CASE WHEN should_go THEN 1 ELSE 0 END) AS should_go_n,
    AVG(CASE WHEN should_go THEN CASE WHEN went_for_it THEN 1.0 ELSE 0.0 END END) AS go_rate_when_should,
    SUM(CASE WHEN fourth_and_one_between_40s THEN 1 ELSE 0 END) AS fourth_and_one_n,
    AVG(
        CASE
            WHEN fourth_and_one_between_40s THEN CASE WHEN went_for_it THEN 1.0 ELSE 0.0 END
        END
    ) AS go_rate_fourth_and_one,
    AVG(wp_left) AS wp_left_per_play,
    AVG(wp_edge) AS mean_edge
FROM decisions
WHERE in_leaderboard
GROUP BY season
ORDER BY season;
