# Feature Engineering Formulas

## India Census 2011 - Engineered Features

This document describes all engineered features created from the India Census 2011 dataset.

---

## Feature List and Formulas

| # | Feature Name | Formula | Description |
|---|-------------|---------|-------------|
| 1 | `sex_ratio` | `(TOT_F / TOT_M) * 1000` | Females per 1000 males |
| 2 | `literacy_rate` | `P_LIT / TOT_P` | Overall literacy rate |
| 3 | `literacy_male_pct` | `M_LIT / TOT_M` | Male literacy rate |
| 4 | `literacy_female_pct` | `F_LIT / TOT_F` | Female literacy rate |
| 5 | `female_literacy_gap` | `literacy_female_pct` | literacy_male_pct |
| 6 | `child_population_ratio` | `P_06 / TOT_P` | Child population (0-6 years) ratio |
| 7 | `sc_pct` | `P_SC / TOT_P` | Scheduled Caste population percentage |
| 8 | `st_pct` | `P_ST / TOT_P` | Scheduled Tribe population percentage |
| 9 | `work_participation_rate` | `TOT_WORK_P / TOT_P` | Overall work participation rate |
| 10 | `work_participation_male_pct` | `TOT_WORK_M / TOT_M` | Male work participation rate |
| 11 | `work_participation_female_pct` | `TOT_WORK_F / TOT_F` | Female work participation rate |
| 12 | `male_female_work_gap` | `work_participation_female_pct` | work_participation_male_pct |
| 13 | `non_worker_pct` | `NON_WORK_P / TOT_P` | Non-worker percentage |
| 14 | `marginal_workers_pct` | `MARGWORK_P / TOT_P` | Marginal workers percentage |
| 15 | `marginal_0_3_vs_3_6_ratio` | `(MARGWORK_0_3_P + 1) / (MARGWORK_3_6_P + 1)` | Ratio of 0-3 to 3-6 month marginal workers |
| 16 | `agri_workers_pct` | `(MAIN_CL_P + MAIN_AL_P) / TOT_WORK_P` | Agricultural workers as % of total workers |
| 17 | `cultivators_pct` | `MAIN_CL_P / TOT_WORK_P` | Cultivators as % of total workers |
| 18 | `agri_labour_pct` | `MAIN_AL_P / TOT_WORK_P` | Agricultural labourers as % of total workers |
| 19 | `household_industry_pct` | `MAIN_HH_P / TOT_WORK_P` | Household industry workers as % of total workers |
| 20 | `other_workers_pct` | `MAIN_OT_P / TOT_WORK_P` | Other workers as % of total workers |
| 21 | `dominant_worker_group` | `argmax(cultivator, agri_labour, household_industry, other)` | Dominant worker category |
| 22 | `urbanisation_rate` | `1 if TRU == "Urban" else 0` | Binary urbanisation indicator |
| 23 | `avg_household_size` | `TOT_P / No_HH` | Average persons per household |
| 24 | `dependency_ratio` | `NON_WORK_P / TOT_WORK_P` | Economic dependency ratio |
| 25 | `child_sex_ratio` | `(F_06 / M_06) * 1000` | Child sex ratio (0-6 years) |
| 26 | `sc_sex_ratio` | `(F_SC / M_SC) * 1000` | SC population sex ratio |
| 27 | `st_sex_ratio` | `(F_ST / M_ST) * 1000` | ST population sex ratio |
| 28 | `illiteracy_rate` | `P_ILL / TOT_P` | Illiteracy rate |
| 29 | `main_to_marginal_ratio` | `MAINWORK_P / MARGWORK_P` | Ratio of main to marginal workers |
| 30 | `pop_per_household` | `TOT_P / No_HH` | Population per household (same as avg_household_size) |

---

## Feature Categories

### Demographics
- `sex_ratio` - Gender balance indicator
- `child_population_ratio` - Youth dependency
- `child_sex_ratio` - Child gender balance

### Literacy
- `literacy_rate` - Overall literacy
- `literacy_male_pct` - Male literacy
- `literacy_female_pct` - Female literacy
- `female_literacy_gap` - Gender literacy disparity
- `illiteracy_rate` - Illiteracy measure

### Work Participation
- `work_participation_rate` - Overall work participation
- `work_participation_male_pct` - Male work participation
- `work_participation_female_pct` - Female work participation
- `male_female_work_gap` - Gender work disparity
- `non_worker_pct` - Non-working population
- `marginal_workers_pct` - Marginal workers
- `marginal_0_3_vs_3_6_ratio` - Marginal worker duration ratio
- `main_to_marginal_ratio` - Main vs marginal workers

### Worker Categories
- `agri_workers_pct` - Agricultural workers
- `cultivators_pct` - Cultivators
- `agri_labour_pct` - Agricultural labourers
- `household_industry_pct` - Household industry
- `other_workers_pct` - Other workers
- `dominant_worker_group` - Primary economic activity (categorical)

### Social Categories
- `sc_pct` - Scheduled Caste percentage
- `st_pct` - Scheduled Tribe percentage
- `sc_sex_ratio` - SC sex ratio
- `st_sex_ratio` - ST sex ratio

### Household
- `avg_household_size` - Persons per household
- `pop_per_household` - Population density indicator
- `dependency_ratio` - Economic dependency

### Urbanisation
- `urbanisation_rate` - Urban indicator (binary)

---

## Notes

1. **Division by Zero**: All divisions are handled safely using `safe_divide()` function
2. **Missing Values**: Missing values are treated as NaN in calculations
3. **Percentage Features**: Features ending in `_pct` or `_rate` are proportions (0-1 scale)
4. **Sex Ratio Features**: Expressed as females per 1000 males

---

*Generated on: 2025-12-11 20:41:18*
