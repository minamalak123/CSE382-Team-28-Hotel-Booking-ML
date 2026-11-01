# Prediction Point and Per-Task Feature Lists

Code: `src/features/feature_lists.py` (single source of truth). Tests: `tests/test_feature_lists.py`.
Generated tables: `outputs/features/` (`column_audit.csv`, `feature_lists.json`, `leakage_evidence.csv`).

## 1. Prediction point

**Booking confirmation:** the moment the hotel confirms a reservation to the customer. We make one prediction per booking.

A column may be used as a feature only if its value is final at that moment. Anything written later (at check-in, at cancellation or during the stay) is excluded from **every** task, because a deployed model would not have it.

| Task | Question answered at confirmation | Target |
|---|---|---|
| Regression | What average daily rate (ADR) will this booking have? | `ADR` |
| Classification | Will this booking be canceled? | `IsCanceled` |
| Clustering | Which booking profile does this booking belong to? | none (`IsCanceled` used only to profile clusters afterwards) |

## 2. Column audit (all 31 raw columns)

| Tier | Columns | Decision |
|---|---|---|
| Leak | `ReservationStatus`, `ReservationStatusDate`, `AssignedRoomType`, `BookingChanges`, `RequiredCarParkingSpaces` | Never used |
| Uncertain | `TotalOfSpecialRequests`, `Country` | Used by default; removed in a strict ablation run |
| Dropped | `ArrivalDateWeekNumber` (redundant with month + day), `Company` (95.4% missing → `HasCompany` flag) | Not used |
| Target | `ADR`, `IsCanceled` | `ADR` is a valid feature for classification/clustering (price agreed at booking); `IsCanceled` is never a feature |
| Safe | All other 22 columns | Used where relevant |

### Evidence (training split only, n = 47,598, base cancel rate 41.7%)

| Check | n in group | Cancel rate in group | Cancel rate otherwise | Verdict |
|---|---:|---:|---:|---|
| `RequiredCarParkingSpaces > 0` | 1,162 | **0.0%** | 42.8% | Leak: recorded on arrival |
| `AssignedRoomType != ReservedRoomType` | 4,356 | 5.8% | 45.4% | Leak: room assigned at check-in |
| `BookingChanges > 0` | 6,146 | 16.6% | 45.5% | Leak: counts changes after booking |
| `ReservationStatus` | – | 1:1 copy of label | – | Leak: it *is* the outcome |
| `TotalOfSpecialRequests > 0` | 18,806 | 21.6% | 54.9% | Uncertain: can be added later |
| `Country == PRT` | 18,630 | 64.7% | 26.9% | Uncertain: often confirmed at check-in |
| `DepositType == Non Refund` | 7,715 | 99.8% | 30.5% | Safe (known at booking), but an anomaly to report |
| `DaysInWaitingList > 0` | 2,058 | 67.9% | 40.6% | Safe: waiting happens *before* confirmation |

## 3. Engineered features (row-wise, from safe columns only)

`TotalNights`, `TotalGuests`, `ArrivalDayOfWeek`, `HasAgent`, `HasCompany`. They learn no statistics from the data, so creating them cannot leak information between splits.

## 4. Feature lists

**Shared block (regression + classification)**
- Numeric (15): `LeadTime`, `ArrivalDateDayOfMonth`, `StaysInWeekendNights`, `StaysInWeekNights`, `Adults`, `Children`, `Babies`, `IsRepeatedGuest`, `PreviousCancellations`, `PreviousBookingsNotCanceled`, `DaysInWaitingList`, `TotalNights`, `TotalGuests`, `HasAgent`, `HasCompany`
- Categorical (9): `ArrivalDateMonth`, `ArrivalDayOfWeek`, `Meal`, `MarketSegment`, `DistributionChannel`, `ReservedRoomType`, `DepositType`, `CustomerType`, `Agent`

| Task | Extra on top of shared block | Total |
|---|---|---|
| Regression (ADR) | `ArrivalDateYear` (price trend: 85.7 → 103.2 → 117.9), `TotalOfSpecialRequests`*, `Country`* | 17 num + 10 cat |
| Classification (IsCanceled) | `ADR`, `TotalOfSpecialRequests`*, `Country`* | 17 num + 10 cat |
| Clustering | Own compact set: `LeadTime`, `TotalNights`, `TotalGuests`, `ADR`, `IsRepeatedGuest`, `PreviousBookingsNotCanceled`, `TotalOfSpecialRequests`* + `MarketSegment`, `CustomerType`, `DepositType` | 7 num + 3 cat |

\* Uncertain tier, removed with `get_feature_lists(task, strict=True)`.

Why clustering uses fewer columns: K-Means measures Euclidean distance, so many one-hot columns from high-cardinality fields (`Country`, `Agent`) would dominate the distances and hide the booking-behaviour structure.

`ArrivalDateYear` is used only for regression because ADR rises every year, while the cancellation rate does not depend on the year (43.9% / 40.5% / 42.4%). Also, 2015 and 2017 are partial years.

## 5. Open items for the leakage audit

1. **Duplicates:** 25,902 duplicate rows. About 5.6k validation and 5.7k test rows are identical to a training row, so the split must be made duplicate-aware before baselines.
2. **Ablation:** train each baseline with `strict=False` and `strict=True` and report the metric gap caused by the uncertain columns.
3. **Non Refund anomaly:** a 99.8% cancel rate for non-refundable deposits is counter-intuitive and is a known issue in this dataset. We keep the column, but it should be discussed in the report.

