-- Teams pooled across graded seasons. Same caution as the coach list:
-- one season is a small sample, eight seasons is a better look.
SELECT
    posteam AS team,
    COUNT(DISTINCT season) AS seasons,
    COUNT(*) AS decisions,
    AVG(CASE WHEN followed THEN 1.0 ELSE 0.0 END) AS follow_rate,
    AVG(CASE WHEN clear_call THEN CASE WHEN followed THEN 1.0 ELSE 0.0 END END) AS follow_rate_clear,
    AVG(wp_left) AS wp_left_per_play,
    STDDEV_SAMP(wp_left) / SQRT(COUNT(*)) AS wp_left_se,
    SUM(wp_left) AS wp_left_total,
    SUM(CASE WHEN should_go THEN 1 ELSE 0 END) AS should_go_n,
    AVG(CASE WHEN should_go THEN CASE WHEN went_for_it THEN 1.0 ELSE 0.0 END END) AS go_rate_when_should
FROM decisions
WHERE in_leaderboard
GROUP BY posteam
HAVING COUNT(*) >= 200
ORDER BY wp_left_per_play;
