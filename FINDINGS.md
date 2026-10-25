# FINDINGS (synthetic data; numbers from `python scripts/compute_findings.py`)

**Setup.** 10 SKUs x 5 regions, daily 2023-2025. Models trained on data before each origin; policies back-tested on 2025 against the true demand with 20 lead-time scenarios (first 28 days = warm-up). Stockout-day rate and stock level of the replayed current policy match real 2025 history (7.15% vs 7.17%; stock +1.2%).

## Headline
| 2025 back-test, network | Current policy | Proposed (95% fill target) | Change |
|---|---|---|---|
| Fill rate (true demand) | 94.6% | 96.8% | **+2.3 pts** |
| Average inventory | IDR 840 m (2.9 days) | IDR 899 m (3.2 days) | +7.0% |
| Lost units / yr | 944 k | 552 k | -42% |
| Total cost (holding + ordering + lost margin) | IDR 3.79 bn/yr | IDR 3.19 bn/yr | **-15.9%** (range -13% to -19% over scenarios) |

The trade-off is explicit: at the same fill rate the proposal needs ~10% less stock; at the same stock it fills +1.5 pts more. A cheaper target (93%) gives +1.3 pts fill at -1.6% stock and -7% cost; 97% gives +3.3 pts at +20% stock and -25% cost.
**Not everywhere:** 11 of 50 stocking points (mostly water, chilled yogurt and Jawa Tengah items, where the current policy already reaches 95-98% fill thanks to short lead times) are not robustly better and stay on the current policy.

## Key findings
1. **Current policy:** 94.6% fill rate and 2.9 inventory days (below the 95% company target). Stockouts concentrate in Lebaran: **36% of 2025 lost dairy-UHT units (64.7 k of 177.7 k) fall in the 12% of days between Ramadan and Lebaran+14**.
2. **Forecast-based policy:** +2.3 pts fill, +7% stock, -16% cost at the company target (see table). 39 of 50 stocking points improve in >=80% of scenarios.
3. **Lead-time variability:** Sumatera lead time is 5.6 days (std 3.3) vs 3.6 (2.3) in Jabodetabek; for the same 95% service it needs **+20% safety stock** (in days of demand). In every region ~63-65% of safety stock exists only because of lead-time variability.
4. **Promotions:** +34% during the 7-day promo, then -9% for 5 days: **19% of the uplift is recovered as a dip, net uplift 16%**.
5. **Censoring:** training on stockout-censored sales under-forecasts true demand by 5.2% (Poisson model) / 4.6% (MA28); excluding stockout days reduces it to -0.8% / -1.1% (oracle -0.05%). The stockout-day fill-rate proxy reads 93.1% vs the true 95.3% (rank correlation of stocking points 0.92).
6. **Forecast accuracy** (50 series, 4 x 28-day rolling origin, pooled WAPE, `scripts/forecast_benchmark.py`): Poisson calendar 18.3%, SARIMAX 18.4%, gradient boosting 19.1%, ETS 20.0%, MA28 20.1%, naive 27.0%. On event days ETS/MA28 reach ~23% vs ~19% for the calendar-aware models; in the Lebaran stress test (window ends 14 Apr 2025) Poisson 19.1%, GBM 20.2%, SARIMAX 20.5% vs ETS 22.2%, MA28 22.7%, naive 29.5%.
7. **Order frequency is the biggest cost block:** ordering = 48% of current cost; the EOQ cycle (about 15-20 days for infant nutrition, 22 for UHT milk) is 2-4x the review period. Longer review periods save an extra simulated IDR 496 m/yr on 28 stocking points without losing fill rate (needs real ordering-cost validation).
8. **Value on the action list:** 67 robust actions, IDR 1.12 bn/yr simulated (621 m replenishment policy + 496 m review period).

## Limitations
One test year / one Lebaran; synthetic world; ordering cost per PO is an assumption of the data; no expiry write-offs, MOQ or truck-load constraints; fill-rate target was not tuned on 2025 (it equals the company target).
