/*
    Warehouse Damage Intelligence
    SQL Server analytics script for the synthetic warehouse_damage table.

    The queries below answer common operational questions about damage cost,
    damage rate, process hotspots, SKU risk, and corrective actions.
*/

/* 1. Preview data */
SELECT TOP 20 *
FROM warehouse_damage
ORDER BY [date] DESC;
GO

/* 2. Executive Summary KPIs */
SELECT
    SUM(cases_handled) AS total_cases_handled,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damaged_cases) * 1.0 / NULLIF(SUM(cases_handled), 0) AS DECIMAL(10, 4)) AS overall_damage_rate,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost,
    SUM(high_damage) AS high_damage_records
FROM warehouse_damage;
GO

/* 3. Damage cost by process area */
SELECT
    process_area,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost
FROM warehouse_damage
GROUP BY process_area
ORDER BY total_damage_cost DESC;
GO

/* 4. Damage rate by process area */
SELECT
    process_area,
    SUM(cases_handled) AS total_cases_handled,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damaged_cases) * 1.0 / NULLIF(SUM(cases_handled), 0) AS DECIMAL(10, 4)) AS damage_rate
FROM warehouse_damage
GROUP BY process_area
ORDER BY damage_rate DESC;
GO

/* 5. Top 10 SKUs by damage cost */
SELECT TOP 10
    sku,
    [description],
    category,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost,
    SUM(damaged_cases) AS total_damaged_cases
FROM warehouse_damage
GROUP BY sku, [description], category
ORDER BY total_damage_cost DESC;
GO

/* 6. Damage rate by category */
SELECT
    category,
    SUM(cases_handled) AS total_cases_handled,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damaged_cases) * 1.0 / NULLIF(SUM(cases_handled), 0) AS DECIMAL(10, 4)) AS damage_rate
FROM warehouse_damage
GROUP BY category
ORDER BY damage_rate DESC;
GO

/* 7. Damage cost by category */
SELECT
    category,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost
FROM warehouse_damage
GROUP BY category
ORDER BY total_damage_cost DESC;
GO

/* 8. Damage cost by aisle */
SELECT
    aisle,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost
FROM warehouse_damage
GROUP BY aisle
ORDER BY total_damage_cost DESC;
GO

/* 9. Damage rate by aisle */
SELECT
    aisle,
    SUM(cases_handled) AS total_cases_handled,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damaged_cases) * 1.0 / NULLIF(SUM(cases_handled), 0) AS DECIMAL(10, 4)) AS damage_rate
FROM warehouse_damage
GROUP BY aisle
ORDER BY damage_rate DESC;
GO

/* 10. Damage source analysis */
SELECT
    damage_source,
    COUNT(*) AS record_count,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost,
    CAST(SUM(damaged_cases) * 1.0 / NULLIF(SUM(cases_handled), 0) AS DECIMAL(10, 4)) AS damage_rate
FROM warehouse_damage
GROUP BY damage_source
ORDER BY total_damage_cost DESC;
GO

/* 11. Equipment type analysis */
SELECT
    equipment_type,
    COUNT(*) AS record_count,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost,
    CAST(SUM(damaged_cases) * 1.0 / NULLIF(SUM(cases_handled), 0) AS DECIMAL(10, 4)) AS damage_rate
FROM warehouse_damage
GROUP BY equipment_type
ORDER BY total_damage_cost DESC;
GO

/* 12. Damage by process area and damage source */
SELECT
    process_area,
    damage_source,
    COUNT(*) AS record_count,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost
FROM warehouse_damage
GROUP BY process_area, damage_source
ORDER BY process_area, total_damage_cost DESC;
GO

/* 13. Operational risk analysis */
SELECT
    congestion_level,
    pallet_stability,
    equipment_condition,
    slotting_risk,
    COUNT(*) AS record_count,
    SUM(cases_handled) AS total_cases_handled,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damaged_cases) * 1.0 / NULLIF(SUM(cases_handled), 0) AS DECIMAL(10, 4)) AS damage_rate,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost
FROM warehouse_damage
GROUP BY
    congestion_level,
    pallet_stability,
    equipment_condition,
    slotting_risk
ORDER BY total_damage_cost DESC, damage_rate DESC;
GO

/* 14. Slot level damage analysis */
SELECT
    slot_level,
    COUNT(*) AS record_count,
    SUM(cases_handled) AS total_cases_handled,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damaged_cases) * 1.0 / NULLIF(SUM(cases_handled), 0) AS DECIMAL(10, 4)) AS damage_rate,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost
FROM warehouse_damage
GROUP BY slot_level
ORDER BY damage_rate DESC;
GO

/* 15. Monthly damage trend */
SELECT
    DATEFROMPARTS(YEAR([date]), MONTH([date]), 1) AS damage_month,
    SUM(cases_handled) AS total_cases_handled,
    SUM(damaged_cases) AS total_damaged_cases,
    CAST(SUM(damaged_cases) * 1.0 / NULLIF(SUM(cases_handled), 0) AS DECIMAL(10, 4)) AS damage_rate,
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS total_damage_cost
FROM warehouse_damage
GROUP BY DATEFROMPARTS(YEAR([date]), MONTH([date]), 1)
ORDER BY damage_month;
GO

/* 16. High-priority action list */
SELECT TOP 20
    sku,
    [description],
    category,
    process_area,
    aisle,
    slot_level,
    damage_source,
    equipment_type,
    congestion_level,
    pallet_stability,
    equipment_condition,
    slotting_risk,
    damaged_cases,
    damage_rate,
    damage_cost,
    CASE
        WHEN damage_cost >= 500 AND damage_rate >= 0.05 THEN 'High'
        WHEN damage_cost >= 250 OR damage_rate >= 0.05 THEN 'Medium'
        ELSE 'Low'
    END AS priority,
    CASE
        WHEN damage_source = 'Driver backing up' THEN 'Review dock safety, backing procedure, and communication controls.'
        WHEN damage_source = 'Freight fell over' THEN 'Review pallet stability, wrapping, and load securement.'
        WHEN damage_source = 'Receiver handling error' THEN 'Review receiving SOP and unloading process.'
        WHEN damage_source = 'Already damaged on arrival' THEN 'Review vendor/carrier claim process and inbound inspection.'
        WHEN damage_source = 'Putaway impact' THEN 'Review slot fit, racking clearance, and MHE handling.'
        WHEN damage_source = 'Replenishment impact' THEN 'Review replenishment path and equipment handling.'
        WHEN damage_source = 'Picker handling' THEN 'Review pick-face setup and product handling standard.'
        WHEN damage_source = 'Shipper loading issue' THEN 'Review loading SOP and pallet build quality.'
        WHEN damage_source = 'Pallet collapse' THEN 'Review stacking pattern and wrap quality.'
        WHEN damage_source = 'Product stacked incorrectly' THEN 'Review heavy-bottom/light-top pallet build standard.'
        ELSE 'Investigate root cause and assign corrective action.'
    END AS recommended_action
FROM warehouse_damage
ORDER BY
    CASE
        WHEN damage_cost >= 500 AND damage_rate >= 0.05 THEN 1
        WHEN damage_cost >= 250 OR damage_rate >= 0.05 THEN 2
        ELSE 3
    END,
    damage_cost DESC,
    damage_rate DESC;
GO

/* 17. Business savings scenarios */
SELECT
    CAST(SUM(damage_cost) AS DECIMAL(14, 2)) AS current_damage_cost,
    CAST(SUM(damage_cost) * 0.10 AS DECIMAL(14, 2)) AS savings_at_10_percent_reduction,
    CAST(SUM(damage_cost) * 0.15 AS DECIMAL(14, 2)) AS savings_at_15_percent_reduction,
    CAST(SUM(damage_cost) * 0.20 AS DECIMAL(14, 2)) AS savings_at_20_percent_reduction
FROM warehouse_damage;
GO
