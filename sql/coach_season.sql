-- Same grading as team_season, for the head coach of the team with the ball.
-- A coach with a small `decisions` count can land at the top or the bottom
-- of a one-season list by noise. The write-up flags seasons under 40
-- decisions and leans on the multi-year table for names.
SELECT
    season,
    coach,
    STRING_AGG(DISTINCT posteam, ', ') AS teams,
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
GROUP BY season, coach
ORDER BY season, wp_left_per_play;
