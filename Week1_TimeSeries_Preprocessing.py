"""
============================================================
INFOTACT INTERNSHIP – PROJECT 2
Supply Chain Analytics: Demand Forecasting & Anomaly Detection
------------------------------------------------------------
WEEK 1: Time-Series Preprocessing & Decomposition
============================================================
Datasets Used:
  1. sales_data.csv            → primary (76,000 rows, 5 categories, daily)
  2. retail_store_inventory.csv → secondary (73,100 rows, Demand Forecast col)
  3. demand_forecasting_dataset.csv → supporting (4,999 rows, future_demand)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose

print("=" * 65)
print("WEEK 1: Time-Series Preprocessing & Decomposition")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# STEP 1: LOAD ALL THREE DATASETS
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading datasets...")

sales = pd.read_csv('/mnt/user-data/uploads/sales_data.csv',
                    parse_dates=['Date'])
inv   = pd.read_csv('/mnt/user-data/uploads/retail_store_inventory.csv',
                    parse_dates=['Date'])
dmd   = pd.read_csv('/mnt/user-data/uploads/demand_forecasting_dataset.csv',
                    parse_dates=['date'])

print(f"  sales_data              : {sales.shape[0]:,} rows x {sales.shape[1]} cols")
print(f"  retail_store_inventory  : {inv.shape[0]:,} rows x {inv.shape[1]} cols")
print(f"  demand_forecasting      : {dmd.shape[0]:,} rows x {dmd.shape[1]} cols")
print(f"  Sales date range        : {sales['Date'].min().date()} → {sales['Date'].max().date()}")
print(f"  Inventory date range    : {inv['Date'].min().date()} → {inv['Date'].max().date()}")
print(f"  Demand date range       : {dmd['date'].min().date()} → {dmd['date'].max().date()}")

# ─────────────────────────────────────────────────────────────────
# STEP 2: CLEAN & SET DATETIME INDEX
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 2] Cleaning and setting datetime index...")

# sales_data cleaning
sales = sales.sort_values('Date').reset_index(drop=True)
sales.columns = [c.strip().lower().replace(' ','_') for c in sales.columns]
print(f"  Missing values in sales_data  : {sales.isnull().sum().sum()}")

# inventory cleaning
inv = inv.sort_values('Date').reset_index(drop=True)
inv.columns = [c.strip().lower().replace(' ','_').replace('/','_') for c in inv.columns]
print(f"  Missing values in inventory   : {inv.isnull().sum().sum()}")

# demand forecasting cleaning
dmd = dmd.sort_values('date').reset_index(drop=True)
print(f"  Missing values in demand_fcst : {dmd.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────
# STEP 3: AGGREGATE DAILY SALES PER CATEGORY
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 3] Aggregating daily sales per category...")

CATEGORIES = sales['category'].unique().tolist()
print(f"  Categories found: {CATEGORIES}")

# Daily units sold per category (across all stores and regions)
daily_cat = (sales.groupby(['date','category'])['units_sold']
             .sum().reset_index()
             .rename(columns={'date':'date','units_sold':'units_sold'}))
daily_cat = daily_cat.set_index('date').sort_index()

# Also aggregate inventory level
daily_inv = (inv.groupby(['date','category'])['units_sold']
             .sum().reset_index()
             .set_index('date').sort_index())

print(f"  Date range in aggregated data : {daily_cat.index.min().date()} → {daily_cat.index.max().date()}")
for cat in CATEGORIES:
    subset = daily_cat[daily_cat['category']==cat]
    print(f"  {cat:15s}: {len(subset):3d} daily records | avg {subset['units_sold'].mean():.0f} units/day")

# ─────────────────────────────────────────────────────────────────
# STEP 4: RESAMPLE TO WEEKLY & MONTHLY + HANDLE MISSING DATES
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 4] Resampling to weekly and monthly aggregates...")

weekly_data  = {}
monthly_data = {}

for cat in CATEGORIES:
    subset = daily_cat[daily_cat['category']==cat]['units_sold'].copy()

    # Create full date range and reindex to fill any missing dates
    full_idx = pd.date_range(start=subset.index.min(),
                              end=subset.index.max(), freq='D')
    subset = subset.reindex(full_idx)
    missing_before = subset.isnull().sum()

    # Interpolate missing dates using time-based interpolation
    subset = subset.interpolate(method='time')
    print(f"  {cat:15s}: {missing_before} missing dates filled via time interpolation")

    weekly_data[cat]  = subset.resample('W').sum()
    monthly_data[cat] = subset.resample('ME').sum()

    # Save cleaned weekly CSV
    weekly_data[cat].to_csv(f'/mnt/user-data/outputs/{cat}_weekly_clean.csv')

print(f"\n  Weekly records per category   : {len(weekly_data[CATEGORIES[0]])} weeks")
print(f"  Monthly records per category  : {len(monthly_data[CATEGORIES[0]])} months")

# ─────────────────────────────────────────────────────────────────
# STEP 5: EDA – OVERVIEW CHART (all 5 categories side by side)
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 5] Generating EDA overview chart...")

COLORS = ['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6']

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle('WEEK 1 EDA — Supply Chain Sales Overview\nInfotact Internship – Project 2',
             fontsize=14, fontweight='bold')

# Chart 1: Daily sales all categories stacked
ax = axes[0, 0]
for i, cat in enumerate(CATEGORIES):
    subset = daily_cat[daily_cat['category']==cat]['units_sold']
    ax.plot(subset.index, subset.values, linewidth=0.8,
            label=cat, color=COLORS[i], alpha=0.8)
ax.set_title('Daily Units Sold by Category', fontweight='bold')
ax.set_ylabel('Units Sold')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
ax.tick_params(axis='x', rotation=25)

# Chart 2: Weekly sales
ax = axes[0, 1]
for i, cat in enumerate(CATEGORIES):
    ax.plot(weekly_data[cat].index, weekly_data[cat].values,
            linewidth=1.5, label=cat, color=COLORS[i])
ax.set_title('Weekly Units Sold by Category', fontweight='bold')
ax.set_ylabel('Units Sold (Weekly Sum)')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=25)

# Chart 3: Monthly total sales bar
ax = axes[0, 2]
monthly_total = sum(monthly_data[cat] for cat in CATEGORIES)
ax.bar(range(len(monthly_total)), monthly_total.values,
       color='#3B82F6', edgecolor='white')
ax.set_title('Monthly Total Units Sold (All Categories)', fontweight='bold')
ax.set_ylabel('Units Sold')
ax.set_xticks(range(len(monthly_total)))
ax.set_xticklabels([d.strftime('%b\n%Y') for d in monthly_total.index], fontsize=7)

# Chart 4: Category share pie
ax = axes[1, 0]
cat_totals = {cat: daily_cat[daily_cat['category']==cat]['units_sold'].sum()
              for cat in CATEGORIES}
ax.pie(cat_totals.values(), labels=cat_totals.keys(),
       autopct='%1.1f%%', colors=COLORS, startangle=90,
       textprops={'fontsize': 9})
ax.set_title('Category Share of Total Units Sold', fontweight='bold')

# Chart 5: Seasonality boxplot
ax = axes[1, 1]
season_order = ['Winter','Spring','Summer','Autumn']
season_data  = [sales[sales['seasonality']==s]['units_sold'].values
                for s in season_order]
bp = ax.boxplot(season_data, labels=season_order, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3B82F6','#10B981','#F59E0B','#EF4444']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Units Sold Distribution by Season', fontweight='bold')
ax.set_ylabel('Units Sold')

# Chart 6: Inventory vs Units Sold
ax = axes[1, 2]
inv_daily = inv.groupby('date')[['units_sold','inventory_level']].mean()
ax.plot(inv_daily.index, inv_daily['inventory_level'],
        color='#3B82F6', linewidth=1, label='Avg Inventory Level')
ax2 = ax.twinx()
ax2.plot(inv_daily.index, inv_daily['units_sold'],
         color='#EF4444', linewidth=1, alpha=0.7, label='Avg Units Sold')
ax.set_title('Inventory Level vs Units Sold Over Time', fontweight='bold')
ax.set_ylabel('Inventory Level', color='#3B82F6')
ax2.set_ylabel('Units Sold', color='#EF4444')
ax.tick_params(axis='x', rotation=25)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=8)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/Week1_EDA_Overview.png',
            dpi=150, bbox_inches='tight')
print("  Saved: Week1_EDA_Overview.png")

# ─────────────────────────────────────────────────────────────────
# STEP 6: TIME SERIES DECOMPOSITION – ALL 5 CATEGORIES
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 6] Running time series decomposition for all 5 categories...")

fig2, big_axes = plt.subplots(5, 4, figsize=(22, 24))
fig2.suptitle('WEEK 1 — Time Series Decomposition (Trend + Seasonality + Residual)\n'
              'All 5 Product Categories | Infotact Internship – Project 2',
              fontsize=13, fontweight='bold', y=1.01)

decomp_results = {}
for row, cat in enumerate(CATEGORIES):
    series = weekly_data[cat].copy()
    series = series.dropna()

    result = seasonal_decompose(series, model='additive', period=52)
    decomp_results[cat] = result

    col_titles = ['Observed (Raw Weekly)', 'Trend Component',
                  'Seasonal Component', 'Residual (Noise)']
    components = [result.observed, result.trend, result.seasonal, result.resid]

    for col, (comp, ctitle) in enumerate(zip(components, col_titles)):
        ax = big_axes[row, col]
        comp.dropna().plot(ax=ax, color=COLORS[row], linewidth=1)
        if row == 0:
            ax.set_title(ctitle, fontweight='bold', fontsize=10)
        if col == 0:
            ax.set_ylabel(cat, fontweight='bold', fontsize=9, color=COLORS[row])
        ax.tick_params(axis='x', rotation=25, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(alpha=0.3)

    print(f"  {cat:15s} decomposed | Trend range: "
          f"{result.trend.dropna().min():.0f}–{result.trend.dropna().max():.0f} | "
          f"Seasonal amplitude: {result.seasonal.max()-result.seasonal.min():.1f}")

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/Week1_Decomposition_AllCategories.png',
            dpi=130, bbox_inches='tight')
print("  Saved: Week1_Decomposition_AllCategories.png")

# ─────────────────────────────────────────────────────────────────
# STEP 7: DETAILED DECOMPOSITION – ELECTRONICS (focus chart)
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 7] Generating detailed decomposition chart (Electronics focus)...")

focus_cat = 'Electronics'
result    = decomp_results[focus_cat]

fig3, axes3 = plt.subplots(4, 1, figsize=(16, 12))
fig3.suptitle(f'Time Series Decomposition — {focus_cat}\n'
              f'Additive Model | Weekly Aggregated | '
              f'Period = 52 weeks',
              fontsize=13, fontweight='bold')

plot_data = [
    (result.observed,  'Observed (Raw Weekly Sales)', '#3B82F6'),
    (result.trend,     'Trend Component',             '#10B981'),
    (result.seasonal,  'Seasonal Component',          '#F59E0B'),
    (result.resid,     'Residual / Noise',            '#EF4444'),
]
for ax, (data, title, color) in zip(axes3, plot_data):
    data.dropna().plot(ax=ax, color=color, linewidth=1.5)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel('Units Sold')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=20)

    # Shade high-season periods
    if 'Seasonal' in title:
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        high = data[data > data.quantile(0.75)]
        for idx in high.index:
            ax.axvspan(idx, idx + pd.Timedelta(weeks=1),
                       alpha=0.15, color='gold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/Week1_Decomposition_Electronics.png',
            dpi=150, bbox_inches='tight')
print("  Saved: Week1_Decomposition_Electronics.png")

# ─────────────────────────────────────────────────────────────────
# STEP 8: DEMAND FORECASTING DATASET – ADDITIONAL EDA
# ─────────────────────────────────────────────────────────────────
print("\n[STEP 8] EDA on demand_forecasting_dataset (future_demand target)...")

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
fig4.suptitle('Week 1 – Demand Forecasting Dataset EDA',
              fontsize=12, fontweight='bold')

# Sales units time series
ax = axes4[0]
dmd_ts = dmd.groupby('date')['sales_units'].sum()
ax.plot(dmd_ts.index, dmd_ts.values, color='#3B82F6', linewidth=1)
ax.set_title('Daily Sales Units\n(demand_forecasting_dataset)', fontweight='bold')
ax.set_ylabel('Sales Units')
ax.tick_params(axis='x', rotation=25)

# Holiday vs Non-holiday
ax = axes4[1]
holiday_avg    = dmd[dmd['holiday_season']==1]['sales_units'].mean()
no_holiday_avg = dmd[dmd['holiday_season']==0]['sales_units'].mean()
ax.bar(['No Holiday','Holiday Season'], [no_holiday_avg, holiday_avg],
       color=['#64748B','#F59E0B'], edgecolor='white')
ax.set_title('Avg Sales: Holiday vs Non-Holiday', fontweight='bold')
ax.set_ylabel('Avg Sales Units')
for i, v in enumerate([no_holiday_avg, holiday_avg]):
    ax.text(i, v+0.5, f'{v:.1f}', ha='center', fontweight='bold')

# Promotion impact
ax = axes4[2]
promo_avg    = dmd[dmd['promotion_applied']==1]['sales_units'].mean()
no_promo_avg = dmd[dmd['promotion_applied']==0]['sales_units'].mean()
ax.bar(['No Promotion','Promotion Applied'], [no_promo_avg, promo_avg],
       color=['#64748B','#10B981'], edgecolor='white')
ax.set_title('Avg Sales: Promotion vs No Promotion', fontweight='bold')
ax.set_ylabel('Avg Sales Units')
for i, v in enumerate([no_promo_avg, promo_avg]):
    ax.text(i, v+0.5, f'{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/Week1_DemandFcst_EDA.png',
            dpi=150, bbox_inches='tight')
print("  Saved: Week1_DemandFcst_EDA.png")

# ─────────────────────────────────────────────────────────────────
# STEP 9: PRINT SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("WEEK 1 SUMMARY")
print("=" * 65)
print(f"  Total rows processed          : {len(sales)+len(inv)+len(dmd):,}")
print(f"  Missing values handled        : 0 (zero missing in all 3 datasets)")
print(f"  Weekly time series created    : {len(CATEGORIES)} (one per category)")
print(f"  Monthly time series created   : {len(CATEGORIES)} (one per category)")
print(f"  Decomposition period          : 52 weeks (annual seasonality)")
print()
for cat in CATEGORIES:
    t = decomp_results[cat].trend.dropna()
    s = decomp_results[cat].seasonal
    print(f"  {cat:15s}: Trend {t.min():.0f}→{t.max():.0f} | "
          f"Seasonal amp {s.max()-s.min():.1f}")
print()
print("  Output files:")
for f in ['Week1_EDA_Overview.png',
          'Week1_Decomposition_AllCategories.png',
          'Week1_Decomposition_Electronics.png',
          'Week1_DemandFcst_EDA.png'] + \
         [f'{c}_weekly_clean.csv' for c in CATEGORIES]:
    print(f"    → {f}")
print("\n  WEEK 1 COMPLETE. Commit all outputs to GitHub.")
