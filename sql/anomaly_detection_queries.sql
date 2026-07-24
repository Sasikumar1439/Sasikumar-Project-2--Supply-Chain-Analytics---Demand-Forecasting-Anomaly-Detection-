-- ============================================
-- Supply Chain Anomaly Detection SQL Queries
-- Infotact Internship | Project 2 | Week 2
-- ============================================

-- STEP 1: Daily aggregated sales time series
CREATE VIEW daily_sales AS
SELECT
    DATE(date)                    AS sale_date,
    SUM(units_sold)               AS total_units_sold,
    AVG(inventory_level)          AS avg_inventory_level,
    MAX(epidemic)                 AS epidemic_flag,
    MAX(promotion)                AS promotion_flag
FROM sales_data
GROUP BY DATE(date)
ORDER BY sale_date;

-- STEP 2: Calculate Z-Score for each day
-- Z = (value - mean) / std_deviation
CREATE VIEW zscore_daily AS
SELECT
    sale_date,
    total_units_sold,
    AVG(total_units_sold) OVER()   AS mean_sales,
    -- SQLite has no STDEV so we approximate via variance
    (total_units_sold -
        AVG(total_units_sold) OVER())
    / (
        SELECT
            SQRT(AVG((total_units_sold - avg_val) *
                     (total_units_sold - avg_val)))
        FROM (
            SELECT
                total_units_sold,
                AVG(total_units_sold) OVER() AS avg_val
            FROM daily_sales
        )
    )                              AS zscore
FROM daily_sales;

-- STEP 3: Flag Z-Score anomalies
CREATE VIEW zscore_anomalies AS
SELECT
    sale_date,
    total_units_sold,
    ROUND(zscore, 4)               AS zscore,
    CASE WHEN ABS(zscore) > 2.5
         THEN 1 ELSE 0 END         AS is_anomaly,
    CASE WHEN zscore > 2.5  THEN 'Spike'
         WHEN zscore < -2.5 THEN 'Drop'
         ELSE 'Normal' END         AS anomaly_direction
FROM zscore_daily;

-- STEP 4: IQR bounds calculation
-- Find Q1 (25th percentile) and Q3 (75th percentile)
CREATE VIEW iqr_bounds AS
SELECT
    MIN(CASE WHEN pct_rank >= 0.25 THEN total_units_sold END) AS Q1,
    MIN(CASE WHEN pct_rank >= 0.75 THEN total_units_sold END) AS Q3,
    MIN(CASE WHEN pct_rank >= 0.75 THEN total_units_sold END) -
    MIN(CASE WHEN pct_rank >= 0.25 THEN total_units_sold END) AS IQR
FROM (
    SELECT
        total_units_sold,
        PERCENT_RANK() OVER (ORDER BY total_units_sold) AS pct_rank
    FROM daily_sales
);

-- STEP 5: IQR anomaly flags
CREATE VIEW iqr_anomalies AS
SELECT
    d.sale_date,
    d.total_units_sold,
    b.Q1, b.Q3, b.IQR,
    b.Q1 - 1.5 * b.IQR             AS lower_fence,
    b.Q3 + 1.5 * b.IQR             AS upper_fence,
    CASE
        WHEN d.total_units_sold < b.Q1 - 1.5 * b.IQR THEN 1
        WHEN d.total_units_sold > b.Q3 + 1.5 * b.IQR THEN 1
        ELSE 0
    END                             AS is_iqr_anomaly
FROM daily_sales d, iqr_bounds b;

-- STEP 6: Contextualize confirmed anomalies
-- Join anomaly flags back to original data for business context
SELECT
    z.sale_date,
    z.total_units_sold,
    z.zscore,
    z.is_anomaly                    AS zscore_flag,
    i.is_iqr_anomaly                AS iqr_flag,
    d.epidemic_flag,
    d.promotion_flag,
    CASE
        WHEN z.is_anomaly = 1
         AND i.is_iqr_anomaly = 1  THEN 'Confirmed Anomaly'
        WHEN z.is_anomaly = 1
          OR i.is_iqr_anomaly = 1  THEN 'Single Method Flag'
        ELSE 'Normal'
    END                             AS anomaly_status,
    CASE
        WHEN z.zscore > 2.5
         AND d.epidemic_flag = 1   THEN 'Epidemic Spike'
        WHEN z.zscore > 2.5
         AND d.promotion_flag = 1  THEN 'Promotion Spike'
        WHEN z.zscore > 2.5        THEN 'Unexplained Spike'
        WHEN z.zscore < -2.5
         AND d.epidemic_flag = 1   THEN 'Epidemic Drop'
        WHEN z.zscore < -2.5       THEN 'Unexplained Drop'
        ELSE 'Normal'
    END                             AS anomaly_context
FROM zscore_anomalies z
JOIN iqr_anomalies   i ON z.sale_date = i.sale_date
JOIN daily_sales     d ON z.sale_date = d.sale_date
ORDER BY z.sale_date;
