# DESCRIPTIVE ANALYTICS & VISUALIZATION AUDIT REPORT
**Dataset:** Forbes Highest-Paid Athletes (1990–2020)  
**Cleaned Dataset:** [`hackathon_live/data/forbes_athletes_cleaned.csv`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/data/forbes_athletes_cleaned.csv)  
**Generator Script:** [`hackathon_live/generate_visualizations.py`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/generate_visualizations.py)  
**Output Visualizations Directory:** [`hackathon_live/visualizations/`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/)  
**Audit Status:** All 8 Audit Verification Checks Completed and **PASSED**  
**Date:** November 21, 2026  

---

## 1. Executive Summary & Analytical Scope

This report documents the descriptive analytics performed on the 30-year Forbes Highest-Paid Athletes dataset ($N = 301$ athlete records, spanning 1990 through 2020). Exactly 11 publication-grade visualizations have been generated, saved as high-resolution (300 DPI) PNG files, and systematically audited against the underlying cleaned data.

### Critical Methodological Disclaimers:
1. **Nominal USD (Unadjusted for Inflation):** All earnings figures are recorded in nominal US Dollars ($ Millions) as published annually by Forbes. They are **not adjusted for inflation** or consumer price index (CPI) changes over the 30-year period. A \$28.6M earnings figure in 1990 represents substantially greater real purchasing power than \$28.6M in 2020.
2. **Distinction Between Aggregate Top-10 Growth, Real Growth, and Athlete Career Growth:**
   - The observed growth in annual Top-10 aggregate earnings (from \$110.8M in 1990 to \$1,059.4M in 2018) reflects the total financial pool captured by the 10 highest-earning spots in those specific years.
   - This must **not** be equated with real economic earnings growth (which is moderated by inflation) or with the career wage growth of individual athletes (as list composition changes each year, reflecting variable athlete performance, contracts, and endorsements).
3. **Cumulative Earnings Scope:** Cumulative earnings represent strictly the **sum of earnings across recorded Forbes Top-10 list appearances in this dataset**, and must **not** be interpreted as lifetime career earnings, pre-list rookie seasons, unranked mid-career years, or post-retirement business equity.
4. **Sport Composition Scope:** Sport percentages represent the proportion of **appearances within Forbes' annual Top-10 rankings**, and do **not** reflect global sports popularity, worldwide athlete participation rates, or total industry revenues.
5. **Sample Size & Reliability Warning for the 2020s Decade ($N = 10$):** The dataset concludes with the year 2020, providing only one year ($N = 10$ observations) for the 2020s decade. While the 1990s ($N = 100$), 2000s ($N = 91$), and 2010s ($N = 100$) represent complete multi-year decades, the 2020s statistics represent an isolated annual snapshot; statistical conclusions for this decade carry substantially higher uncertainty and are less reliable for longitudinal comparison.
6. **Sparse Categories & Extreme Outlier Distortion:**
   - Cycling ($N=1$, Lance Armstrong \$28M) and MMA ($N=1$, Conor McGregor \$99M) are singletons. Baseball ($N=6$) and Hockey ($N=4$) have minimal samples. Their statistics reflect isolated contracts/fights rather than representative sport-wide distributions.
   - Boxing contains extreme right-tail outliers (Floyd Mayweather: \$300M in 2015, \$285M in 2018; Manny Pacquiao: \$160M in 2015), which heavily inflate the sport's mean and variance relative to its median.
7. **Calendar Irregularities:** Forbes omitted publication in 2001 ($N=0$), and published 11 athletes in 2002 due to an exact tie for Rank 10 between Jacques Villeneuve and Jeff Gordon (\$20.0M each).

---

## 2. Verified Summary Tables of Descriptive Statistics

### Table A: Overall Earnings Distribution ($N = 301$ Records)
All figures verified against `hackathon_live/data/forbes_athletes_cleaned.csv`:

| Descriptive Statistic | Value ($ Millions Nominal USD) | Data Characteristics & Distribution Interpretation |
| :--- | :---: | :--- |
| **Record Count ($N$)** | 301 | Total athlete-year observations across 30 published lists |
| **Mean ($\mu$)** | \$45.52M | Elevated relative to median due to extreme right-tail observations |
| **Standard Deviation ($\sigma$)** | \$33.53M | Substantial dispersion across eras, sports, and ranking positions |
| **Minimum** | \$8.10M | Evander Holyfield, Arnold Palmer, Michael Jordan (1990) |
| **25th Percentile ($Q_1$)** | \$24.00M | Lower quartile threshold of Top-10 earnings |
| **50th Percentile (Median)** | \$39.00M | Robust central tendency of Top-10 earnings |
| **75th Percentile ($Q_3$)** | \$59.40M | Upper quartile threshold of Top-10 earnings |
| **Interquartile Range ($IQR$)**| \$35.40M | Middle 50% spread of recorded appearances |
| **Maximum** | \$300.00M | Floyd Mayweather Jr. (2015, single-year prize purse and PPV revenue) |
| **Skewness** | **+3.10** | Strongly right-skewed; majority earn under \$60M with sparse high-value tail |
| **Kurtosis** | **+18.62** | Extreme leptokurtic profile (fat positive tail compared to a Gaussian distribution) |

---

### Table B: Per-Decade Historical Statistics
| Decade | Records ($N$) | Total Nominal Earnings ($M) | Mean ($M) | Median ($M) | Min ($M) | Max ($M) | Std Dev ($M) | Reliability Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1990s** | 100 | \$2,276.6M | \$22.77M | \$17.10M | \$8.1M | \$78.3M | \$14.80M | Complete 10-year sample (1990–1999) |
| **2000s** | 91 | \$3,694.1M | \$40.59M | \$35.00M | \$20.0M | \$115.0M | \$19.98M | 9 published years (omits 2001; 11 rows in 2002) |
| **2010s** | 100 | \$6,910.5M | \$69.10M | \$59.45M | \$32.3M | \$300.0M | \$39.33M | Complete 10-year sample (2010–2019) |
| **2020s** | 10 | \$819.2M | \$81.92M | \$81.30M | \$59.1M | \$106.3M | \$19.96M | **Single-year sample ($N=10$); high sampling variance** |

---

### Table C: Per-Sport Compensation & Sample Size Breakdown
| Canonical Sport (`Sport_Clean`) | Appearances ($N$) | Dataset Share (%) | Median ($M) | Mean ($M) | Std Dev ($M) | Min ($M) | Max ($M) | Statistical Stability Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Basketball** | 82 | 27.24% | \$36.00M | \$41.87M | \$20.20M | \$8.1M | \$89.0M | Highly stable; large multi-decade sample |
| **Boxing** | 46 | 15.28% | \$30.75M | \$49.97M | \$59.64M | \$8.1M | \$300.0M | High variance; heavily influenced by Mayweather/Pacquiao outliers |
| **Golf** | 44 | 14.62% | \$48.25M | \$48.56M | \$30.20M | \$8.1M | \$115.0M | Stable sample; Tiger Woods peak observed at \$115.0M |
| **Auto Racing** | 40 | 13.29% | \$29.00M | \$32.02M | \$18.58M | \$9.0M | \$80.0M | Stable sample; includes F1, NASCAR, and MotoGP |
| **Soccer** | 33 | 10.96% | \$64.70M | \$67.38M | \$30.29M | \$28.0M | \$127.0M | High median; concentrated primarily in the 2010s/2020 |
| **Tennis** | 23 | 7.64% | \$44.50M | \$44.23M | \$28.62M | \$8.5M | \$106.3M | Moderate sample; includes Agassi, Sampras, Federer, Nadal, Djokovic |
| **American Football** | 21 | 6.98% | \$43.80M | \$45.77M | \$21.59M | \$9.5M | \$89.5M | Moderate sample; includes quarterback contracts and Deion Sanders |
| **Baseball** | 6 | 1.99% | \$26.10M | \$24.35M | \$8.10M | \$12.7M | \$35.0M | **Sparse sample ($N=6$); interpret with caution** |
| **Hockey** | 4 | 1.33% | \$16.20M | \$18.92M | \$7.49M | \$13.5M | \$29.8M | **Sparse sample ($N=4$); interpret with caution** |
| **Cycling** | 1 | 0.33% | \$28.00M | \$28.00M | N/A | \$28.0M | \$28.0M | **Singleton ($N=1$, Lance Armstrong 2006); no distribution** |
| **MMA** | 1 | 0.33% | \$99.00M | \$99.00M | N/A | \$99.0M | \$99.0M | **Singleton ($N=1$, Conor McGregor 2018); no distribution** |

---

### Table D: Rank vs. Earnings Progression
| Current Rank | Observations ($N$) | Mean Earnings ($M) | Median Earnings ($M) | Min ($M) | Max ($M) | Mean Drop from Previous Rank ($M) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 30 | \$91.40M | \$79.30M | \$28.6M | \$300.0M | Baseline |
| **2** | 30 | \$61.38M | \$53.65M | \$16.7M | \$160.0M | -\$30.02M (-32.8%) |
| **3** | 30 | \$49.35M | \$44.25M | \$13.0M | \$108.0M | -\$12.03M (-19.6%) |
| **4** | 30 | \$44.16M | \$38.75M | \$10.0M | \$99.0M | -\$5.19M (-10.5%) |
| **5** | 30 | \$38.59M | \$32.70M | \$9.0M | \$93.4M | -\$5.57M (-12.6%) |
| **6** | 30 | \$38.96M | \$36.00M | \$8.6M | \$89.5M | +\$0.37M (+1.0%) |
| **7** | 30 | \$34.19M | \$29.10M | \$8.5M | \$89.3M | -\$4.77M (-12.2%) |
| **8** | 30 | \$32.37M | \$28.90M | \$8.1M | \$89.0M | -\$1.82M (-5.3%) |
| **9** | 30 | \$31.88M | \$29.00M | \$9.0M | \$79.8M | -\$0.49M (-1.5%) |
| **10** | 31 *(tie in 2002)* | \$31.33M | \$26.10M | \$8.5M | \$65.4M | -\$0.55M (-1.7%) |

---

## 3. Audited In-Depth Analysis of the 11 Visualizations

---

### Plot 1: Earnings Distribution Histogram & KDE
- **Saved File:** [`hackathon_live/visualizations/plot_01_earnings_distribution.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_01_earnings_distribution.png)
- **Analytical Purpose:** Characterize the frequency distribution, central tendency, and tail behavior of nominal athlete earnings across the 30-year dataset.
- **Actual Observation:**
  - Distribution exhibits a marked positive skew ($\text{skewness} = +3.10$) and strong leptokurtic shape ($\text{kurtosis} = +18.62$).
  - Approximately 75% of observations fall between \$8.1M and \$59.4M, with a modal peak near the median of \$39.0M.
  - A sparse, elongated right tail reaches up to \$300.0M, primarily populated by isolated mega-events (Mayweather in 2015 and 2018).
  - The sample mean (\$45.52M) sits \$6.52M above the sample median (\$39.0M), reflecting the upward pull of these right-tail values.
- **Interpretation:** The empirical distribution is consistent with observations in labor economics where compensation at the extreme upper echelon of entertainment markets displays heavy right-tail concentration.
- **Limitations:** Combines observations spanning 30 different years in nominal dollars without adjusting for inflation, which partially conflates true high-earner premiums with historical price-level increases.

---

### Plot 2: Earnings Boxplot by Cleaned Sport
- **Saved File:** [`hackathon_live/visualizations/plot_02_earnings_by_sport.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_02_earnings_by_sport.png)
- **Analytical Purpose:** Compare compensation spreads, medians, interquartile ranges, and outlier occurrences across canonical sports.
- **Actual Observation:**
  - Among sports with substantial sample sizes ($N \ge 20$), **Soccer** displays the highest sample median (\$64.70M, $N=33$), followed by **Golf** (\$48.25M, $N=44$), **American Football** (\$43.80M, $N=21$), **Tennis** (\$44.50M, $N=23$), **Basketball** (\$36.00M, $N=82$), **Boxing** (\$30.75M, $N=46$), and **Auto Racing** (\$29.00M, $N=40$).
  - **Boxing** displays the widest dispersion ($\sigma = \$59.64M, \text{Max} = \$300.0M$), driven by isolated event paydays, while its median (\$30.75M) ranks below most team sports.
  - **Basketball** exhibits the largest number of observations ($N=82$) with moderate spread ($IQR = \$26.98M$).
- **Interpretation:** Different sports exhibit differing compensation structures: team sports with salary caps (e.g. NBA, NFL) show narrower relative spreads among their elite, whereas individual pay-per-view sports (Boxing) produce episodic spikes.
- **Limitations:** Cycling ($N=1$) and MMA ($N=1$) represent single observations and provide no measure of dispersion. Baseball ($N=6$) and Hockey ($N=4$) have small sample sizes and should not be used to infer broader sport-wide compensation.

---

### Plot 3: Year-Wise Total Earnings Line Plot
- **Saved File:** [`hackathon_live/visualizations/plot_03_yearwise_total_earnings.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_03_yearwise_total_earnings.png)
- **Analytical Purpose:** Track the longitudinal trajectory of aggregate nominal earnings for the Top 10 athletes per year from 1990 to 2020.
- **Actual Observation:**
  - Aggregate annual earnings of the Top 10 rose from **\$110.8M in 1990** to a peak of **\$1,059.4M in 2018**, ending at **\$819.2M in 2020**.
  - A steady upward trajectory is visible from 1990 to 2000 (\$110.8M $\to$ \$341.0M), followed by a plateau between 2002 and 2007 (\$307M–\$343M), and accelerated growth post-2010 with pronounced spikes in 2015 (\$952.0M) and 2018 (\$1,059.4M).
  - The line is discontinuous across 2001 due to the absence of a Forbes publication in that year.
- **Interpretation:** The upward trajectory in nominal aggregate earnings coincides historically with the expansion of international television rights, global corporate endorsements, and digital broadcasting revenues.
- **Limitations:** Figures reflect nominal USD. Part of the aggregate growth represents general macroeconomic inflation over 30 years rather than pure expansion in real purchasing power. Furthermore, this measures the aggregate of the top 10 positions, not the career trajectories of individual athletes.

---

### Plot 4: Year-Wise Athlete Record Count Line Plot
- **Saved File:** [`hackathon_live/visualizations/plot_04_yearwise_record_count.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_04_yearwise_record_count.png)
- **Analytical Purpose:** Verify record completeness per calendar year and document dataset structural anomalies.
- **Actual Observation:**
  - Exactly 10 records appear in 28 of the 30 published years.
  - **Year 2001 has 0 records:** Forbes did not publish an athlete earnings list in 2001.
  - **Year 2002 has 11 records:** Caused by an exact tie for Rank #10 between Jacques Villeneuve and Jeff Gordon (\$20.0M each).
- **Interpretation:** Validates the structural integrity of the dataset and confirms why total rows equal $301$ ($28 \times 10 + 11 + 10 = 301$).
- **Limitations:** Record counts reflect Forbes' editorial truncation (Top 10 cutoff), providing no visibility into athletes ranked 11 or lower.

---

### Plot 5: Sport-Wise Record Count Bar Chart
- **Saved File:** [`hackathon_live/visualizations/plot_05_sport_record_count.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_05_sport_record_count.png)
- **Analytical Purpose:** Measure the frequency of appearances captured by each canonical sport in Forbes' annual Top-10 lists.
- **Actual Observation:**
  - **Basketball:** 82 appearances (27.24% of dataset).
  - **Boxing:** 46 appearances (15.28%).
  - **Golf:** 44 appearances (14.62%).
  - **Auto Racing:** 40 appearances (13.29%).
  - **Soccer:** 33 appearances (10.96%).
  - **Tennis:** 23 appearances (7.64%).
  - **American Football:** 21 appearances (6.98%).
  - Minor categories: Baseball (6, 1.99%), Hockey (4, 1.33%), Cycling (1, 0.33%), MMA (1, 0.33%).
- **Interpretation:** The four leading sports account for **70.43%** of all historical Top-10 positions. Basketball's high frequency is associated with multi-year career longevity of key superstars (Jordan, O'Neal, Bryant, James) and major endorsement deals.
- **Limitations:** Frequencies represent appearances on Forbes' annual Top-10 list, **not** overall sports popularity, worldwide athlete counts, or participation rates.

---

### Plot 6: Nationality Distribution Bar Chart (Top Categories + Other)
- **Saved File:** [`hackathon_live/visualizations/plot_06_nationality_distribution.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_06_nationality_distribution.png)
- **Analytical Purpose:** Characterize the geographic distribution of athlete entries across the dataset.
- **Actual Observation:**
  - **USA accounts for 206 of 301 records (68.44%).**
  - The second tier consists of UK (13, 4.32%), Germany (13, 4.32%), Switzerland (12, 3.99%), Portugal (10, 3.32%), Brazil (9, 2.99%), and Argentina (9, 2.99%).
  - The remaining 14 nationalities combined represent 29 records (9.63%).
- **Interpretation:** Historical Top-10 appearances are heavily concentrated among athletes from the United States, associated with the financial scale of domestic US sports leagues (NBA, NFL) and US-hosted commercial events. Top European soccer players (Messi, Ronaldo) and tennis stars (Federer) account for the majority of top international positions.
- **Limitations:** Grouping 14 nations into an "Other" category aggregates diverse countries (Philippines, Canada, Italy, France, etc.) that each have small appearance counts ($1$ to $4$).

---

### Plot 7: Current Rank vs. Earnings Scatterplot
- **Saved File:** [`hackathon_live/visualizations/plot_07_rank_vs_earnings.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_07_rank_vs_earnings.png)
- **Analytical Purpose:** Examine the relationship between list ranking and annual earnings across eras.
- **Actual Observation:**
  - Rank #1 displays a substantially higher sample mean (\$91.40M) and median (\$79.30M) than any other rank position.
  - A steep downward slope is observed from Rank #1 to Rank #3:
    - Rank #2 mean: \$61.38M (-\$30.02M relative to Rank #1).
    - Rank #3 mean: \$49.35M (-\$12.03M relative to Rank #2).
  - Across Ranks #4 through #10, the gradient flattens considerably (means range from \$44.16M down to \$31.33M).
  - Modern era records (2010s in red, 2020s in purple) sit systematically higher on the vertical axis across all rank positions compared to 1990s and 2000s records.
- **Interpretation:** The relationship is consistent with superstar economics models (Rosen, 1981): the apex earner in any given year exhibits a substantial earnings premium over ranks #2 and #3, while ranks #4 through #10 display a more compressed spread.
- **Limitations:** Because ranks are constructed by sorting athletes by earnings within each year, Rank and Earnings are mechanically dependent by design within each annual cohort.

---

### Plot 8: Earnings Distribution by Decade Boxplot
- **Saved File:** [`hackathon_live/visualizations/plot_08_earnings_by_decade.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_08_earnings_by_decade.png)
- **Analytical Purpose:** Compare the central tendencies and dispersions of nominal earnings across the four calendar decades.
- **Actual Observation:**
  - **1990s ($N=100$):** Median = \$17.10M, IQR = \$11.50M, Mean = \$22.77M ($\sigma = \$14.80M$).
  - **2000s ($N=91$):** Median = \$35.00M, IQR = \$16.00M, Mean = \$40.59M ($\sigma = \$19.98M$). Median increased by 104.7% over 1990s.
  - **2010s ($N=100$):** Median = \$59.45M, IQR = \$32.80M, Mean = \$69.10M ($\sigma = \$39.33M$). Median increased by 69.9% over 2000s; dispersion widened considerably.
  - **2020s ($N=10$):** Median = \$81.30M, IQR = \$29.40M, Mean = \$81.92M ($\sigma = \$19.96M$).
- **Interpretation:** The empirical medians and quartiles shift steadily upward across successive decades in nominal terms, alongside an expansion in interquartile spread.
- **Limitations:** **Crucial reliability limitation:** The 2020s sample contains only 10 observations (the 2020 list), which means its statistics represent a single point-in-time cross-section rather than a decade-long trend, carrying higher sampling uncertainty than the preceding 10-year samples.

---

### Plot 9: Correlation Matrix Heatmap
- **Saved File:** [`hackathon_live/visualizations/plot_09_correlation_heatmap.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_09_correlation_heatmap.png)
- **Analytical Purpose:** Quantify pairwise linear associations among numerical and ordinal features.
- **Actual Observation:**
  - **`Year` vs. `Earnings` ($r = +0.65$):** Computed at the **individual observation level ($N=301$)**, indicating a moderate-to-strong positive linear association between publication calendar year and an athlete's recorded earnings.
  - **`Current Rank` vs. `Earnings` ($r = -0.45$):** Computed at the **individual observation level ($N=301$)**, confirming a moderate negative linear association (as rank index increases from 1 to 10, nominal earnings decrease).
  - **`Rank_Tier_Ordinal` vs. `Earnings` ($r = -0.47$):** Consistent with Current Rank.
  - **`Decade_Ordinal` vs. `Earnings` ($r = +0.60$):** Reinforces the decade-level shift observed in Plot 8.
  - **`Nationality_Is_USA` vs. `Earnings` ($r = -0.13$):** Weak negative linear association.
- **Observation Level & Aggregation Clarification:**
  - The row-level correlation ($N=301$) between `Year` and `Earnings` is **$r = +0.65$**.
  - When computed at the **annual aggregate level ($N=30$ years)**, the Pearson correlation between `Year` and `Total Annual Top-10 Earnings` is **$r = +0.92$** (and $r = +0.92$ for `Mean Annual Top-10 Earnings`). The row-level correlation is naturally lower ($0.65$) because it includes intra-year variation between ranks 1 through 10.
  - The negative sign of $r_{\text{Rank, Earnings}} = -0.45$ is expected because Rank 1 represents the highest earnings. Because athletes within each year are ordered by earnings, the relationship is monotonic within each year; across the pooled 30-year dataset, the correlation is attenuated from $-1.0$ to $-0.45$ due to cross-era wage inflation (e.g. Rank 10 in 2020 earned \$59.1M, exceeding Rank 1 in 1990 at \$28.6M).
- **Limitations:** Pearson correlation captures strictly linear associations and does not quantify non-linear curvature.

---

### Plot 10: Top Athletes by Cumulative Recorded Earnings
- **Saved File:** [`hackathon_live/visualizations/plot_10_top_athletes_cumulative_earnings.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_10_top_athletes_cumulative_earnings.png)
- **Analytical Purpose:** Identify individual athletes with the highest aggregate nominal earnings across their recorded Forbes Top-10 list appearances.
- **Actual Observation:**
  - **Tiger Woods:** \$1,373.8M across 19 recorded appearances (Golf, 1997–2020; average \$72.3M per appearance).
  - **LeBron James:** \$844.8M across 13 recorded appearances (Basketball, 2008–2020; average \$65.0M per appearance).
  - **Floyd Mayweather Jr.:** \$840.0M across 5 recorded appearances (Boxing, 2010–2018; average \$168.0M per appearance).
  - **Cristiano Ronaldo:** \$787.1M across 10 recorded appearances (Soccer, 2011–2020; average \$78.7M per appearance).
  - **Roger Federer:** \$781.1M across 12 recorded appearances (Tennis, 2008–2020; average \$65.1M per appearance).
  - **Michael Jordan:** \$738.8M across 19 recorded appearances (Basketball, 1990–2009; average \$38.9M per appearance).
  - **Lionel Messi:** \$715.5M across 9 recorded appearances (Soccer, 2011–2020; average \$79.5M per appearance).
  - **Michael Schumacher:** \$639.0M across 13 recorded appearances (Auto Racing, 1995–2011; average \$49.2M per appearance).
  - **Kobe Bryant:** \$601.1M across 14 recorded appearances (Basketball, 2002–2016; average \$42.9M per appearance).
  - **Phil Mickelson:** \$519.9M across 11 recorded appearances (Golf, 2006–2016; average \$47.3M per appearance).
- **Interpretation:** Illustrates two distinct profiles associated with high cumulative recorded earnings:
  1. *High Longevity:* Athletes appearing across 10 to 19 years (Tiger Woods, Michael Jordan, Kobe Bryant, LeBron James).
  2. *High Peak Event Paydays:* Athletes appearing fewer times but generating massive single-year paydays (Floyd Mayweather generated \$840.0M across only 5 appearances).
- **CRITICAL SCOPE DISCLAIMER:** These sums represent **only earnings during years the athlete appeared in Forbes' annual Top-10 rankings**. They do **not** reflect lifetime career income, earnings outside the top 10, pre-peak seasons, or business equity post-retirement.

---

### Plot 11: Sport Composition Dynamics Across Decades (Stacked Chart)
- **Saved File:** [`hackathon_live/visualizations/plot_11_sport_composition_by_decade.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_11_sport_composition_by_decade.png)
- **Analytical Purpose:** Track the longitudinal shift in the proportion of Top-10 spots occupied by different sports across decades.
- **Actual Observation:**
  - **1990s ($N=100$):** Dominated by Basketball (30.0%), Boxing (24.0%), Auto Racing (16.0%), and Golf (11.0%). Soccer accounted for 0.0% of Top-10 spots.
  - **2000s ($N=91$):** Golf expanded to 24.18%, Auto Racing to 23.08%, Basketball held 23.08%, and Soccer emerged with 7.69% (Beckham, Ronaldinho).
  - **2010s ($N=100$):** Basketball held 28.00%, while **Soccer expanded substantially to 22.00%** (Ronaldo, Messi, Neymar). Boxing represented 13.00%, while Auto Racing fell to 3.00%.
  - **2020s ($N=10$):** Basketball held 30.0%, Soccer 30.0%, American Football 20.0%, Tennis 10.0%, and Golf 10.0%. Boxing and Auto Racing had 0 entries in 2020.
- **Interpretation:** Illustrates an observable shift in the sports represented among the world's 10 highest-paid athletes: individual combat and motorsports held higher shares in the 1990s/2000s, whereas globally televised soccer and basketball stars occupied larger shares in the 2010s and 2020.
- **Limitations:** Proportions reflect representation on Forbes' annual Top-10 list, **not** worldwide sports popularity or participation. The 2020s data represents a single year ($N=10$), making it sensitive to single-year contract fluctuations.

---

## 4. Verification Table of Visualization Artifacts

All 11 visualization files have been verified to exist on disk as valid, non-empty PNG images:

| Plot # | Filename | File Size | Pixel Dimensions | Verification Status |
| :---: | :--- | :---: | :---: | :---: |
| **1** | [`plot_01_earnings_distribution.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_01_earnings_distribution.png) | 217.7 KB | 3000 $\times$ 1800 | **VERIFIED (Non-empty)** |
| **2** | [`plot_02_earnings_by_sport.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_02_earnings_by_sport.png) | 265.4 KB | 3600 $\times$ 2100 | **VERIFIED (Non-empty)** |
| **3** | [`plot_03_yearwise_total_earnings.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_03_yearwise_total_earnings.png) | 269.1 KB | 3600 $\times$ 1800 | **VERIFIED (Non-empty)** |
| **4** | [`plot_04_yearwise_record_count.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_04_yearwise_record_count.png) | 212.7 KB | 3300 $\times$ 1500 | **VERIFIED (Non-empty)** |
| **5** | [`plot_05_sport_record_count.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_05_sport_record_count.png) | 220.1 KB | 3300 $\times$ 1800 | **VERIFIED (Non-empty)** |
| **6** | [`plot_06_nationality_distribution.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_06_nationality_distribution.png) | 188.0 KB | 3000 $\times$ 1800 | **VERIFIED (Non-empty)** |
| **7** | [`plot_07_rank_vs_earnings.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_07_rank_vs_earnings.png) | 421.6 KB | 3000 $\times$ 1800 | **VERIFIED (Non-empty)** |
| **8** | [`plot_08_earnings_by_decade.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_08_earnings_by_decade.png) | 160.1 KB | 2700 $\times$ 1800 | **VERIFIED (Non-empty)** |
| **9** | [`plot_09_correlation_heatmap.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_09_correlation_heatmap.png) | 249.5 KB | 2400 $\times$ 1950 | **VERIFIED (Non-empty)** |
| **10** | [`plot_10_top_athletes_cumulative_earnings.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_10_top_athletes_cumulative_earnings.png) | 311.4 KB | 3300 $\times$ 2100 | **VERIFIED (Non-empty)** |
| **11** | [`plot_11_sport_composition_by_decade.png`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/plot_11_sport_composition_by_decade.png) | 175.2 KB | 3300 $\times$ 1950 | **VERIFIED (Non-empty)** |
