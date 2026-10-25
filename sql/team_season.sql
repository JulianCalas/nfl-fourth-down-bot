-- Win probability left on the table by the team that had the ball.
-- One row per team per season. `wp_left_se` is the ordinary standard error
-- of the per-play mean. Plays in the same game are not independent, so the
-- interval is a little too narrow. Treat it as a sample-size warning, not
-- as a precise confidence interval.
SELECT
    season,
    posteam AS team,
    COUNT(*) AS decisions,
    COUNT(DISTINCT game_id) AS games,
    AVG(CASE WHEN followed THEN 1.0 ELSE 0.0 END) AS follow_rate,
    AVG(CASE WHEN clear_call THEN CASE WHEN followed THEN 1.0 ELSE 0.0 END END) AS follow_rate_clear,
    AVG(wp_left) AS wp_left_per_play,
    STDDEV_SAMP(wp_left) / SQRT(COUNT(*)) AS wp_left_se,
    SUM(wp_left) AS wp_left_total,
    SUM(CASE WHEN should_go THEN 1 ELSE 0 END) AS should_go_n,
    AVG(CASE WHEN should_go THEN CASE WHEN went_for_it THEN 1.0 ELSE 0.0 END END) AS go_rate_when_should
FROM decisions
WHERE in_leaderboard
GROUP BY season, posteam
ORDER BY season, wp_left_per_play;
