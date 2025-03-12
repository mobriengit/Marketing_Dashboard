-- Marketing Dashboard SQL Queries
-- These queries are designed to be used with the processed marketing campaign data
-- and expanded marketing data for visualization in Tableau or other BI tools.

-- 1. Overall Campaign Performance
SELECT 
    campaign_name,
    SUM(spend) AS total_spend,
    SUM(impressions) AS total_impressions,
    SUM(clicks) AS total_clicks,
    SUM(conversions) AS total_conversions,
    SUM(revenue) AS total_revenue,
    (SUM(clicks) / SUM(impressions)) * 100 AS ctr_percentage,
    SUM(spend) / SUM(clicks) AS cpc,
    SUM(spend) / SUM(conversions) AS cpa,
    SUM(revenue) / SUM(spend) AS roas
FROM 
    processed_campaign_data
GROUP BY 
    campaign_name
ORDER BY 
    total_revenue DESC;

-- 2. Channel Performance Comparison
SELECT 
    channel,
    SUM(spend) AS total_spend,
    SUM(impressions) AS total_impressions,
    SUM(clicks) AS total_clicks,
    SUM(conversions) AS total_conversions,
    SUM(revenue) AS total_revenue,
    (SUM(clicks) / SUM(impressions)) * 100 AS ctr_percentage,
    SUM(spend) / SUM(clicks) AS cpc,
    SUM(spend) / SUM(conversions) AS cpa,
    SUM(revenue) / SUM(spend) AS roas
FROM 
    processed_campaign_data
GROUP BY 
    channel
ORDER BY 
    roas DESC;

-- 3. Monthly Performance Trends
SELECT 
    year_month,
    SUM(spend) AS total_spend,
    SUM(impressions) AS total_impressions,
    SUM(clicks) AS total_clicks,
    SUM(conversions) AS total_conversions,
    SUM(revenue) AS total_revenue,
    (SUM(clicks) / SUM(impressions)) * 100 AS ctr_percentage,
    SUM(spend) / SUM(clicks) AS cpc,
    SUM(spend) / SUM(conversions) AS cpa,
    SUM(revenue) / SUM(spend) AS roas
FROM 
    processed_campaign_data
GROUP BY 
    year_month
ORDER BY 
    year_month;

-- 4. Campaign Efficiency Analysis
SELECT 
    campaign_name,
    channel,
    SUM(spend) AS total_spend,
    SUM(budget) AS total_budget,
    (SUM(spend) / SUM(budget)) * 100 AS budget_utilization,
    SUM(revenue) / SUM(spend) AS roas,
    SUM(spend) / SUM(conversions) AS cpa
FROM 
    processed_campaign_data
GROUP BY 
    campaign_name, channel
ORDER BY 
    roas DESC;

-- 5. Daily Performance Metrics
SELECT 
    campaign_name,
    channel,
    AVG(daily_spend) AS avg_daily_spend,
    AVG(daily_impressions) AS avg_daily_impressions,
    AVG(daily_clicks) AS avg_daily_clicks,
    AVG(daily_conversions) AS avg_daily_conversions,
    AVG(daily_revenue) AS avg_daily_revenue
FROM 
    processed_campaign_data
GROUP BY 
    campaign_name, channel
ORDER BY 
    avg_daily_revenue DESC;

-- 6. Campaign Duration Impact Analysis
SELECT 
    campaign_duration,
    COUNT(*) AS campaign_count,
    AVG(ctr) * 100 AS avg_ctr_percentage,
    AVG(cpc) AS avg_cpc,
    AVG(cpa) AS avg_cpa,
    AVG(roas) AS avg_roas
FROM 
    processed_campaign_data
GROUP BY 
    campaign_duration
ORDER BY 
    campaign_duration;

-- 7. Top Performing Campaigns by ROAS
SELECT 
    campaign_name,
    channel,
    SUM(revenue) AS total_revenue,
    SUM(spend) AS total_spend,
    SUM(revenue) / SUM(spend) AS roas
FROM 
    processed_campaign_data
GROUP BY 
    campaign_name, channel
ORDER BY 
    roas DESC
LIMIT 10;

-- 8. Worst Performing Campaigns by ROAS
SELECT 
    campaign_name,
    channel,
    SUM(revenue) AS total_revenue,
    SUM(spend) AS total_spend,
    SUM(revenue) / SUM(spend) AS roas
FROM 
    processed_campaign_data
GROUP BY 
    campaign_name, channel
ORDER BY 
    roas ASC
LIMIT 10;

-- 9. Quarterly Performance Analysis
SELECT 
    year,
    quarter,
    SUM(spend) AS total_spend,
    SUM(impressions) AS total_impressions,
    SUM(clicks) AS total_clicks,
    SUM(conversions) AS total_conversions,
    SUM(revenue) AS total_revenue,
    (SUM(clicks) / SUM(impressions)) * 100 AS ctr_percentage,
    SUM(spend) / SUM(clicks) AS cpc,
    SUM(spend) / SUM(conversions) AS cpa,
    SUM(revenue) / SUM(spend) AS roas
FROM 
    processed_campaign_data
GROUP BY 
    year, quarter
ORDER BY 
    year, quarter;

-- 10. Budget Utilization Analysis
SELECT 
    campaign_name,
    SUM(budget) AS total_budget,
    SUM(spend) AS total_spend,
    (SUM(spend) / SUM(budget)) * 100 AS budget_utilization_percentage,
    CASE 
        WHEN (SUM(spend) / SUM(budget)) * 100 < 80 THEN 'Under-utilized'
        WHEN (SUM(spend) / SUM(budget)) * 100 BETWEEN 80 AND 95 THEN 'Optimally utilized'
        ELSE 'Over-utilized'
    END AS budget_utilization_category
FROM 
    processed_campaign_data
GROUP BY 
    campaign_name
ORDER BY 
    budget_utilization_percentage DESC;

-- EXPANDED MARKETING DATA QUERIES

-- 11. Ad Platform Performance Comparison
SELECT 
    ad_platform,
    SUM(ad_spend) AS total_spend,
    SUM(ad_clicks) AS total_clicks,
    SUM(ad_impressions) AS total_impressions,
    (SUM(ad_clicks) / SUM(ad_impressions)) * 100 AS ctr_percentage,
    AVG(ad_conversion_rate) * 100 AS avg_conversion_rate,
    AVG(ad_CPC) AS avg_cpc
FROM 
    processed_expanded_data
GROUP BY 
    ad_platform
ORDER BY 
    ctr_percentage DESC;

-- 12. Social Platform Engagement Analysis
SELECT 
    social_platform,
    SUM(social_engagements) AS total_engagements,
    SUM(social_reach) AS total_reach,
    (SUM(social_engagements) / SUM(social_reach)) * 100 AS engagement_rate
FROM 
    processed_expanded_data
GROUP BY 
    social_platform
ORDER BY 
    engagement_rate DESC;

-- 13. Email Campaign Performance
SELECT 
    email_subject,
    SUM(email_recipients) AS total_recipients,
    SUM(email_opens) AS total_opens,
    SUM(email_clicks) AS total_clicks,
    (SUM(email_opens) / SUM(email_recipients)) * 100 AS open_rate,
    (SUM(email_clicks) / SUM(email_opens)) * 100 AS click_through_rate,
    AVG(email_bounce_rate) * 100 AS avg_bounce_rate,
    AVG(email_unsub_rate) * 100 AS avg_unsubscribe_rate
FROM 
    processed_expanded_data
GROUP BY 
    email_subject
ORDER BY 
    click_through_rate DESC;

-- 14. Cross-Channel Performance by Campaign
SELECT 
    campaign_id,
    ad_platform,
    social_platform,
    email_subject,
    SUM(ad_clicks) AS total_ad_clicks,
    SUM(social_engagements) AS total_social_engagements,
    SUM(email_clicks) AS total_email_clicks,
    AVG(ad_CTR) * 100 AS avg_ad_ctr,
    AVG(social_CTR) * 100 AS avg_social_ctr,
    AVG(email_click_rate) * 100 AS avg_email_ctr,
    AVG(campaign_effectiveness) AS campaign_effectiveness_score
FROM 
    processed_expanded_data
GROUP BY 
    campaign_id, ad_platform, social_platform, email_subject
ORDER BY 
    campaign_effectiveness_score DESC;

-- 15. Monthly Cross-Channel Performance
SELECT 
    year_month,
    SUM(ad_spend) AS total_ad_spend,
    SUM(ad_clicks) AS total_ad_clicks,
    SUM(ad_impressions) AS total_ad_impressions,
    SUM(social_engagements) AS total_social_engagements,
    SUM(social_reach) AS total_social_reach,
    SUM(email_clicks) AS total_email_clicks,
    SUM(email_opens) AS total_email_opens,
    SUM(email_recipients) AS total_email_recipients,
    (SUM(ad_clicks) / SUM(ad_impressions)) * 100 AS ad_ctr,
    (SUM(social_engagements) / SUM(social_reach)) * 100 AS social_engagement_rate,
    (SUM(email_opens) / SUM(email_recipients)) * 100 AS email_open_rate,
    (SUM(email_clicks) / SUM(email_opens)) * 100 AS email_click_rate
FROM 
    processed_expanded_data
GROUP BY 
    year_month
ORDER BY 
    year_month;

-- 16. Ad Platform and Social Platform Correlation
SELECT 
    ad_platform,
    social_platform,
    COUNT(*) AS campaign_count,
    AVG(ad_CTR) * 100 AS avg_ad_ctr,
    AVG(social_CTR) * 100 AS avg_social_ctr,
    AVG(ad_to_social_ratio) AS avg_ad_to_social_ratio,
    AVG(campaign_effectiveness) AS avg_effectiveness_score
FROM 
    processed_expanded_data
GROUP BY 
    ad_platform, social_platform
ORDER BY 
    avg_effectiveness_score DESC;

-- 17. Email and Ad Performance Correlation
SELECT 
    ad_platform,
    email_subject,
    COUNT(*) AS campaign_count,
    AVG(ad_CTR) * 100 AS avg_ad_ctr,
    AVG(email_click_rate) * 100 AS avg_email_ctr,
    AVG(email_to_ad_ratio) AS avg_email_to_ad_ratio,
    AVG(campaign_effectiveness) AS avg_effectiveness_score
FROM 
    processed_expanded_data
GROUP BY 
    ad_platform, email_subject
ORDER BY 
    avg_effectiveness_score DESC;

-- 18. Campaign Effectiveness Ranking
SELECT 
    campaign_id,
    ad_platform,
    social_platform,
    email_subject,
    AVG(campaign_effectiveness) AS effectiveness_score,
    RANK() OVER (ORDER BY AVG(campaign_effectiveness) DESC) AS effectiveness_rank
FROM 
    processed_expanded_data
GROUP BY 
    campaign_id, ad_platform, social_platform, email_subject
ORDER BY 
    effectiveness_rank;

-- 19. Channel Contribution Analysis
SELECT 
    campaign_id,
    (AVG(ad_CTR) * 100 * 0.3) AS ad_contribution,
    (AVG(social_CTR) * 100 * 0.3) AS social_contribution,
    (AVG(email_click_rate) * 100 * 0.4) AS email_contribution,
    AVG(campaign_effectiveness) * 100 AS total_effectiveness
FROM 
    processed_expanded_data
GROUP BY 
    campaign_id
ORDER BY 
    total_effectiveness DESC;

-- 20. Combined Campaign Performance (if datasets can be joined)
-- This is a placeholder query assuming the datasets can be joined on campaign_id
-- In practice, you would need to adjust this based on how the datasets are actually related
/*
SELECT 
    c.campaign_id,
    c.campaign_name,
    e.ad_platform,
    e.social_platform,
    e.email_subject,
    c.spend AS campaign_spend,
    c.revenue AS campaign_revenue,
    c.roas AS campaign_roas,
    e.ad_CTR * 100 AS ad_ctr_percentage,
    e.social_CTR * 100 AS social_ctr_percentage,
    e.email_click_rate * 100 AS email_ctr_percentage,
    e.campaign_effectiveness * 100 AS effectiveness_score
FROM 
    processed_campaign_data c
JOIN 
    processed_expanded_data e ON c.campaign_id = e.campaign_id
ORDER BY 
    effectiveness_score DESC;
*/ 