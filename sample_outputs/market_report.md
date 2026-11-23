## Executive Summary
This is an **offline demonstration report** for the topic *Tech Sector Q3 Analysis*. It was produced without
any LLM or web access from a bundled illustrative dataset of 10 fictional companies,
so it contains **no real market data**. Run the crew without `--offline` (and with API keys
configured) to generate a real report.

Across the sample, mean revenue is 30.26 bn USD and the mean
operating margin is 20.20%.

## Market Overview
The illustrative sector contains cloud, semiconductor, software, hardware, analytics and security
vendors. Scale differs widely: revenue ranges from 6.90 to
81.30 bn USD, which is reflected in a standard deviation of
23.11.

## Key Financial Metrics
The table below is produced by the Pandas-based Financial Data Parser tool.

| metric | count | mean | std | min | max |
| --- | --- | --- | --- | --- | --- |
| revenue_usd_bn | 10 | 30.26 | 23.11 | 6.90 | 81.30 |
| net_income_usd_bn | 10 | 6.05 | 5.63 | -0.80 | 17.60 |
| operating_margin_pct | 10 | 20.20 | 10.73 | -6.20 | 31.50 |
| yoy_growth_pct | 10 | 17.13 | 17.72 | 3.40 | 64.50 |
| rd_spend_usd_bn | 10 | 4.54 | 2.65 | 1.70 | 9.80 |

## Analysis & Insights
- **Largest company by revenue:** Orbitron Devices (81.3 bn USD).
- **Fastest growth:** Aurora AI Labs (64.5% year over year).
- **Profitability spread:** operating margin spans -6.2% to
  31.5%, so margin quality differs far more than headline growth.
- **R&D intensity:** mean R&D spend is 4.54 bn USD per company.

## Risks & Limitations
- The dataset is illustrative and synthetic; conclusions must not be used for investment decisions.
- Sample size is small (10 rows), so standard deviations are unstable.
- Numeric consistency checks run on the data:

- Rows: 10, columns: 6.
- No missing values.
- No duplicate rows.
- Non-numeric (label) columns: company.
- Column 'net_income_usd_bn' has 1 negative value(s) - verify they are valid.
- Column 'operating_margin_pct' has 1 negative value(s) - verify they are valid.

## Conclusion
The pipeline (research, analysis, writing, PDF compilation, benchmarking) works end to end. Replace the
offline dataset with live research output by running with valid API keys.

## Sources
- Bundled file `data/sample_metrics.csv` (synthetic, for demonstration only).
