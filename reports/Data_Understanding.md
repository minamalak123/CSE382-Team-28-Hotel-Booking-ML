# Hotel Booking ML Project — Complete Dataset Understanding & Validation Summary

## 1. Dataset Overview

### Dataset

- **File:** `H2.csv`
- **Rows:** 79,330
- **Columns:** 31
- **Dataset type:** Hotel booking records
- **Main ML tasks:**
  1. **Classification:** Predict `IsCanceled`
  2. **Regression:** Predict `ADR`

The validation script loads the raw dataset from:

```text
data/raw/H2.csv
```

The script verifies the dataset structure, missing values, duplicates, target validity, numerical constraints, dates, cancellation consistency, and booking-level logical consistency.

---

# 2. The Most Important Idea: What Does One Row Represent?

The safest interpretation from our current data is:

> **One row represents one hotel booking/reservation record.**

Each row contains information about:

- when the guest planned to arrive
- how far in advance they booked
- how many guests were involved
- how long they planned to stay
- where the customer came from
- how the booking was made
- room information
- previous booking history
- special requests
- price (`ADR`)
- final reservation status
- whether the booking was canceled

Therefore, our dataset is not simply "customer information."

It is a combination of:

```text
Booking information
        +
Guest information
        +
Stay information
        +
Customer history
        +
Distribution/channel information
        +
Room information
        +
Price information
        +
Final booking outcome
```

---

# 3. The 31 Columns

## A. Target / Outcome Columns

### 1. `IsCanceled`

**Meaning:** Whether the booking was canceled.

Values:

```text
0 = not canceled
1 = canceled
```

Our validation found:

```text
Unique values: [0, 1]
Invalid values: 0
```

So the target is valid as a binary classification target.

### Important

For our classification task:

```text
X = booking features
y = IsCanceled
```

The model should learn:

> Given information about a booking, what is the probability that it will be canceled?

---

### 2. `ADR`

**ADR = Average Daily Rate**

This is the price/rate associated with the booking on a daily basis.

For our project, `ADR` is the **regression target**.

Current statistics:

| Statistic          |     ADR |
| ------------------ | ------: |
| Count              |  79,330 |
| Mean               | 105.304 |
| Standard deviation |  43.603 |
| Minimum            |       0 |
| 25%                |    79.2 |
| Median             |    99.9 |
| 75%                |     126 |
| Maximum            |    5400 |



Therefore:

```text
Regression:
X = booking-related features
y = ADR
```

The model should learn:

> Given information about a booking, what ADR should we expect?

---

# 4. Arrival / Date Information

These columns describe **when the guest is expected to arrive**.

## 3. `ArrivalDateYear`

Year of arrival.

Observed data contains years such as:

```text
2015
2016
2017
```

---

## 4. `ArrivalDateMonth`

Month of arrival.

Examples:

```text
January
February
March
...
December
```

This is stored as a string.

---

## 5. `ArrivalDateWeekNumber`

Week number of the arrival date.

Validation rule:

```text
1 <= week number <= 53
```

Result:

```text
Invalid week numbers: 0
```

---

## 6. `ArrivalDateDayOfMonth`

Day of the month on which the guest arrives.

Validation rule:

```text
1 <= day <= 31
```

Result:

```text
Invalid day numbers: 0
```

---

## Date validation

We performed two levels of validation.

### Level 1 — Component validation

We checked:

- month names
- week numbers
- day numbers

Results:

```text
Invalid months: 0
Invalid weeks: 0
Invalid days: 0
```



### Level 2 — Full calendar validation

We converted:

```text
Year + Month + Day
```

into an actual calendar date.

Result:

```text
Invalid or incomplete arrival dates: 0
```



### Conclusion

The arrival-date components are currently **calendar-valid**.

---

# 5. Stay Information

These columns describe how long the guest planned to stay.

## 7. `StaysInWeekendNights`

Number of nights during the weekend.

Example:

```text
2
```

means two weekend nights.

---

## 8. `StaysInWeekNights`

Number of nights during weekdays.

Example:

```text
5
```

means five weekday nights.

---

## Total stay length

We can derive:

```text
TotalNights =
    StaysInWeekendNights
    +
    StaysInWeekNights
```

This is an important feature we can potentially engineer later.

Example:

```text
Weekend nights = 2
Week nights = 5

Total nights = 7
```

---

# 6. Guest Information

## 9. `Adults`

Number of adults in the booking.

Validation:

```text
Negative values: 0
```

---

## 10. `Children`

Number of children.

Important:

```text
Data type = float64
```

This is because the column contains missing values.

Missing:

```text
4 rows
```

---

## 11. `Babies`

Number of babies.

Validation:

```text
Negative values: 0
Missing values: 0
```

---

## Important derived feature

We can potentially create:

```text
TotalGuests =
    Adults + Children + Babies
```

But we need to decide how to handle the 4 missing `Children` values first.

---

# 7. Booking Timing

## 12. `LeadTime`

Number of days between booking and arrival.

Conceptually:

```text
Booking date
      |
      | LeadTime days
      ↓
Arrival date
```

Example:

```text
LeadTime = 100
```

means the booking was made approximately 100 days before arrival.

Validation:

```text
Negative values: 0
```

This could be a very important predictor for cancellation.

---

# 8. Meal Information

## 13. `Meal`

Meal plan associated with the booking.

This is categorical data.

Examples from the dataset include values such as:

```text
BB
HB
```

Because this is categorical, we should not automatically treat the strings as numerical values.

During preprocessing, it will require categorical encoding.

---

# 9. Customer Origin

## 14. `Country`

Country associated with the customer/booking.

This is categorical data.

There are:

```text
24 missing values
```



We need to decide how to handle these during preprocessing.

---

# 10. Booking Acquisition / Distribution

## 15. `MarketSegment`

Describes the market segment through which the booking was made.

This is categorical.

Potentially useful for understanding:

> What type of customer/market generated the booking?

---

## 16. `DistributionChannel`

Describes the distribution channel used for the booking.

This is also categorical.

Potentially useful for cancellation prediction because different booking channels may have different cancellation behavior.

---

# 11. Customer History

## 17. `IsRepeatedGuest`

Indicates whether the customer is a repeated guest.

Conceptually:

```text
0 = not repeated
1 = repeated
```

This is an important behavioral feature.

---

## 18. `PreviousCancellations`

Number of previous canceled bookings associated with the customer.

Example:

```text
PreviousCancellations = 3
```

means the customer had three previous cancellations.

This may be highly relevant for predicting future cancellation.

---

## 19. `PreviousBookingsNotCanceled`

Number of previous bookings that were not canceled.

This gives us another measure of customer history.

Potential relationship:

```text
Previous cancellations
+
Previous successful bookings
```

can provide information about customer behavior.

---

# 12. Room Information

## 20. `ReservedRoomType`

Room type originally reserved.

Categorical feature.

---

## 21. `AssignedRoomType`

Room type actually assigned.

Categorical feature.

### Important distinction

These are not necessarily the same.

For example:

```text
ReservedRoomType = A
AssignedRoomType = B
```

could indicate that the customer reserved one room type but was assigned another.

The difference between these columns may itself contain useful information.

We may later create:

```text
RoomChanged =
    ReservedRoomType != AssignedRoomType
```

But this should be considered feature engineering, not something we should automatically do before understanding the data.

---

# 13. Booking Modification Information

## 22. `BookingChanges`

Number of changes made to the booking.

Example:

```text
BookingChanges = 0
```

means no recorded booking changes.

Higher values indicate more modifications.

This could potentially help predict cancellation behavior.

---

# 14. Deposit Information

## 23. `DepositType`

Type of deposit associated with the booking.

Categorical feature.

This can potentially be important for cancellation prediction because deposit policies may affect cancellation behavior.

---

# 15. Agent / Company

## 24. `Agent`

Identifier associated with the booking agent.

This is categorical / identifier-like data.

### Important

We should **not immediately assume that Agent is a normal numerical feature**.

Even if it appears numerically represented, it represents categories/identifiers rather than a quantity.

For ML:

```text
Agent 1
Agent 2
Agent 3
```

does NOT mean:

```text
Agent 3 > Agent 2 > Agent 1
```

---

## 25. `Company`

Company associated with the booking.

Also identifier-like / categorical.

It may have many distinct values and therefore requires careful preprocessing.

---

# 16. Waiting List

## 26. `DaysInWaitingList`

Number of days the booking was on the waiting list.

Validation:

```text
Negative values: 0
```

This is numerical/count data.

---

# 17. Customer Type

## 27. `CustomerType`

Type of customer.

Categorical feature.

It can potentially represent different customer behaviors.

---

# 18. Parking and Special Requests

## 28. `RequiredCarParkingSpaces`

Number of car parking spaces requested.

Validation:

```text
Negative values: 0
```

---

## 29. `TotalOfSpecialRequests`

Number of special requests.

Validation:

```text
Negative values: 0
```

This is a count feature.

It could represent customer engagement or complexity of the booking.

---

# 19. Final Reservation Outcome

## 30. `ReservationStatus`

Final status of the reservation.

Observed values:

| ReservationStatus |  Count |
| ----------------- | -----: |
| Check-Out         | 46,228 |
| Canceled          | 32,186 |
| No-Show           |    916 |



---

# 20. Reservation Status vs IsCanceled

This is one of the most important validations we performed.

The relationship is:

| `IsCanceled` | ReservationStatus |  Count |
| -----------: | ----------------- | -----: |
|            0 | Check-Out         | 46,228 |
|            1 | Canceled          | 32,186 |
|            1 | No-Show           |    916 |

There are no cases where:

```text
IsCanceled = 0 + Canceled
```

or

```text
IsCanceled = 0 + No-Show
```

The cross-tabulation confirms the consistency.

### Therefore

For this dataset:

```text
IsCanceled = 0
    → Check-Out

IsCanceled = 1
    → Canceled OR No-Show
```

---

# 21. Very Important ML Warning: Target Leakage

`ReservationStatus` is extremely closely connected to `IsCanceled`.

Therefore, if our goal is:

```text
Predict IsCanceled
```

we must carefully consider whether `ReservationStatus` should be used as an input feature.

The model could essentially learn:

```text
ReservationStatus = Canceled
→ IsCanceled = 1
```

That would not represent meaningful prediction.

This is a classic example of **target leakage**.

The same principle applies to any feature that contains information only known after the cancellation/outcome occurred.

We therefore need to define:

> **What information would actually be available at prediction time?**

before finalizing the feature set.

---

# 22. `ReservationStatusDate`

## 31. `ReservationStatusDate`

Date associated with the final reservation status.

It is currently stored as:

```text
str
```

This is potentially important because it contains temporal information.

However, we must be careful with it.

If we are predicting cancellation **before the booking outcome happens**, the final status date may contain future information.

Therefore:

```text
Do NOT automatically use it as a normal feature.
```

We need to determine whether it is available at prediction time.

---

# 23. Data Types

Current types include:

### Integer

Examples:

```text
IsCanceled
LeadTime
ArrivalDateYear
ArrivalDateWeekNumber
ArrivalDateDayOfMonth
Adults
Babies
PreviousCancellations
BookingChanges
DaysInWaitingList
...
```

### Float

```text
Children
ADR
```

`Children` is float because of missing values.

### String / categorical

Examples:

```text
ArrivalDateMonth
Meal
Country
MarketSegment
DistributionChannel
ReservedRoomType
AssignedRoomType
DepositType
Agent
Company
CustomerType
ReservationStatus
ReservationStatusDate
```

The exact types are shown in the validation output.

---

# 24. Missing Values

The dataset has only:

```text
28 rows containing missing values
```

Missing values:

| Column   | Missing |
| -------- | ------: |
| Children |       4 |
| Country  |      24 |

Everything else:

```text
0 missing
```



### Important conclusion

The dataset is **not heavily affected by missing values**.

28 / 79,330 rows is extremely small relative to the full dataset.

However, we still need a documented strategy.

Possible options:

```text
Children → impute
Country → categorical "Unknown"
```

But the final strategy should be decided during preprocessing rather than silently changing the raw dataset.

---

# 25. Duplicate Rows

This is currently one of the biggest data-quality issues.

The dataset contains:

```text
25,902 exact duplicate rows
```

There are:

```text
31,748 rows belonging to duplicate groups
```

If exact duplicates were removed:

```text
79,330
→ 53,428 rows
```



---

# 26. What Does "Duplicate Rows" Mean?

An exact duplicate means:

> Every value across all 31 columns is identical.

Example:

```text
Row A
IsCanceled = 1
LeadTime = 62
ArrivalDateYear = 2015
...
ADR = 76.5
ReservationStatus = No-Show

Row B
IsCanceled = 1
LeadTime = 62
ArrivalDateYear = 2015
...
ADR = 76.5
ReservationStatus = No-Show
```

If every column matches, pandas considers them exact duplicates.

The validation output shows examples of exactly repeated records.

---

# 27. Important: Do NOT Automatically Delete Duplicates Yet

This needs a team decision.

We currently know:

```text
25,902 exact duplicate rows exist.
```

But we do not yet know whether:

1. They are accidental duplicates.
2. They represent repeated booking records.
3. They are inherent in the source dataset.
4. Removing them would distort the real distribution.

Therefore:

> **We should investigate and document the duplicate policy before deleting them.**

Especially because removing duplicates changes the dataset from:

```text
79,330 → 53,428
```

which is a substantial change.

---

# 28. ADR Investigation

`ADR` has an unusual distribution.

Current findings:

```text
ADR = 0 → 1,208 rows
ADR > 500 → 2 rows
Maximum ADR → 5,400
```



---

# 29. ADR = 0

There are:

```text
1,208 rows
```

with:

```text
ADR = 0
```

Breakdown:

```text
IsCanceled = 0 → 1,079
IsCanceled = 1 → 129
```



These are not automatically errors.

A zero ADR may have a legitimate interpretation, but we need to understand the source/business meaning before deciding whether to:

- keep
- remove
- transform
- treat as a special case

---

# 30. ADR Outliers

Only:

```text
2 rows have ADR > 500
```

The highest is:

```text
ADR = 5400
```

Another is:

```text
ADR = 510
```



The `5400` value is especially suspicious because it is much larger than:

```text
Median = 99.9
Mean ≈ 105.3
```

This is a strong outlier.

But:

> **Outlier ≠ automatically incorrect data.**

We should investigate before removing or transforming it.

---

# 31. ADR Distribution We Need to Understand

Current summary:

```text
Minimum = 0
Q1 = 79.2
Median = 99.9
Q3 = 126
Maximum = 5400
```

This tells us:

- most observations are much closer to the 79–126 range
- the maximum is extremely far away
- the distribution is likely strongly right-skewed
- regression models may be affected by extreme values

This is something EDA should investigate visually.

---

# 32. Zero-ADR Records and Guest Information

Among the 1,208 zero-ADR records, the most common guest combinations include:

```text
1 adult, 0 children, 0 babies → 582
2 adults, 0 children, 0 babies → 412
0 adults, 0 children, 0 babies → 137
```



This is interesting because some zero-ADR records also have unusual guest counts.

Therefore zero ADR should be investigated together with:

```text
Adults
Children
Babies
Stays
IsCanceled
ReservationStatus
```

rather than looking at ADR alone.

---

# 33. Missing Values + Duplicates Interaction

This is important.

Some of the missing-value rows appear repeatedly.

For example, several records have:

```text
LeadTime = 193
Adults = 2
Children = 0
Country = missing
ADR = 65
IsCanceled = 1
ReservationStatus = Canceled
```

Therefore, when we investigate missing values, we should also consider whether some missing-value records are part of duplicate groups.

Do not assume:

```text
28 missing rows = 28 independent cases
```

because duplicates exist in the dataset.

---

# 34. Numerical Validity

We checked whether count-like features contain negative values.

Checked columns:

```text
LeadTime
StaysInWeekendNights
StaysInWeekNights
Adults
Children
Babies
PreviousCancellations
PreviousBookingsNotCanceled
BookingChanges
DaysInWaitingList
RequiredCarParkingSpaces
TotalOfSpecialRequests
```

Every one returned:

```text
0 negative values
```



### Conclusion

There are no negative values in the checked count variables.

---

# 35. Target Validity

## Classification target

```text
IsCanceled
```

Valid values:

```text
0
1
```

Invalid:

```text
0 rows
```

## Regression target

`ADR` was checked for:

```text
negative values
non-finite values
```

Result:

```text
0 invalid rows
```



### Important distinction

ADR has unusual values:

```text
0
5400
```

but they are not technically invalid according to the current validation rule.

Therefore:

```text
VALID ≠ NORMAL
```

A value can be technically valid but statistically unusual.

---

# 36. Booking Logical Consistency

We checked three important conditions.

## Zero adults

```text
390 bookings
```

have:

```text
Adults = 0
```

## Zero total nights

```text
331 bookings
```

have:

```text
StaysInWeekendNights + StaysInWeekNights = 0
```

## Both zero adults AND zero nights

```text
67 bookings
```

have both conditions.



---

# 37. Why Are These Important?

A normal hotel booking would generally be expected to have:

```text
At least one guest
+
At least one night
```

But our dataset contains records such as:

```text
Adults = 0
Children = 0
Babies = 0
```

and/or:

```text
Total nights = 0
```

Some examples are associated with:

```text
ReservationStatus = Check-Out
```



These records therefore require investigation.

### Important

We should **not delete them automatically**.

They may represent:

- unusual booking situations
- data-entry behavior
- special cases
- no-stay bookings
- source-dataset characteristics

We need to understand them first.

---

# 38. What We Have Proven So Far

Our current validation gives us the following picture:

| Area                                | Result                      |
| ----------------------------------- | --------------------------- |
| Dataset size                        | 79,330 × 31                 |
| Missing values                      | 28 rows                     |
| Duplicate rows                      | 25,902 exact duplicate rows |
| Invalid `IsCanceled`                | 0                           |
| Invalid/non-finite ADR              | 0                           |
| Negative count values               | 0                           |
| Invalid arrival months              | 0                           |
| Invalid arrival weeks               | 0                           |
| Invalid arrival days                | 0                           |
| Invalid complete arrival dates      | 0                           |
| Cancellation/status inconsistencies | 0                           |
| Zero ADR                            | 1,208                       |
| ADR > 500                           | 2                           |
| Zero adults                         | 390                         |
| Zero nights                         | 331                         |
| Zero adults + zero nights           | 67                          |

---

# 39. The Most Important Things We Need to Understand Before Modeling

There are several different categories of data issues.

## Category 1 — Clearly valid

Examples:

```text
No negative LeadTime
Valid arrival dates
Valid IsCanceled values
Valid numerical ADR values
Consistent ReservationStatus
```

These do not currently require correction.

---

## Category 2 — Missing data

```text
Children → 4 missing
Country → 24 missing
```

Needs a preprocessing strategy.

---

## Category 3 — Duplicates

```text
25,902 exact duplicate rows
```

Needs investigation and a documented decision.

---

## Category 4 — Statistical outliers

```text
ADR = 5400
ADR = 510
```

Needs EDA and domain reasoning.

---

## Category 5 — Potentially unusual business records

```text
Adults = 0
Total nights = 0
```

Needs logical/business investigation.

---

## Category 6 — Potential target leakage

Most importantly:

```text
ReservationStatus
ReservationStatusDate
```

may contain information about the final outcome.

These must be reviewed before modeling.

---

# 40. Raw Data vs Clean Data

We should maintain this distinction throughout the project.

## Raw data

```text
data/raw/H2.csv
```

This should remain unchanged.

It represents the original dataset.

---

## Cleaned data

After we decide:

- duplicate policy
- missing-value strategy
- outlier strategy
- invalid/unusual record strategy
- feature engineering

we can produce a processed dataset.

Conceptually:

```text
Raw Dataset
     ↓
Validation
     ↓
EDA
     ↓
Cleaning Decisions
     ↓
Preprocessing
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Model Training
```

---

# 41. Very Important: Validation vs EDA vs Preprocessing

These three are different.

## Validation

Question:

> Is the data structurally and logically valid?

Examples:

```text
Are dates valid?
Are counts negative?
Is IsCanceled only 0/1?
Are there missing values?
Are there duplicates?
```

We have already done a large portion of this.

---

## EDA

Question:

> What patterns and distributions exist in the data?

Examples:

```text
Which features correlate with cancellation?
How does ADR vary?
Which months have more cancellations?
Does LeadTime affect cancellation?
Which market segments cancel more?
```

EDA is where we start understanding the behavior of the dataset.

---

## Preprocessing

Question:

> How do we transform the data into a form suitable for ML?

Examples:

```text
Handle missing values
Encode categorical features
Scale numerical features when appropriate
Remove or treat selected outliers
Engineer useful features
Separate X and y
Split training/testing data
```

---

# 42. Two Separate ML Problems

Our project is effectively handling two supervised learning tasks.

# Classification

```text
Target:
IsCanceled
```

Goal:

```text
Predict whether a booking will be canceled.
```

Potential evaluation metrics:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
```

We should not rely only on accuracy.

---

# Regression

```text
Target:
ADR
```

Goal:

```text
Predict the expected Average Daily Rate.
```

Potential evaluation metrics:

```text
MAE
MSE
RMSE
R²
```

Because ADR contains extreme values, we should pay attention to how outliers affect regression metrics.

---

# 43. Feature vs Target

A very important concept for the team:

## Target

What we want the model to predict.

```text
Classification:
y = IsCanceled

Regression:
y = ADR
```

## Features

Information used by the model to make the prediction.

```text
X = selected booking information
```

---

# 44. We Should Not Use Everything Automatically

Having 31 columns does **not** mean we should feed all 31 columns into the model.

For each column, we need to ask:

### Question 1

Is it available at prediction time?

### Question 2

Is it a feature or the target?

### Question 3

Does it leak the target?

### Question 4

Is it numerical or categorical?

### Question 5

Does it need preprocessing?

### Question 6

Does it contain too many unique categories?

### Question 7

Does it represent an identifier rather than meaningful numerical information?

---

# 45. Preliminary Feature Categories

## Numerical / count features

Examples:

```text
LeadTime
StaysInWeekendNights
StaysInWeekNights
Adults
Children
Babies
PreviousCancellations
PreviousBookingsNotCanceled
BookingChanges
DaysInWaitingList
ADR
RequiredCarParkingSpaces
TotalOfSpecialRequests
```

---

## Date/time features

```text
ArrivalDateYear
ArrivalDateMonth
ArrivalDateWeekNumber
ArrivalDateDayOfMonth
ReservationStatusDate
```

These may require date feature engineering.

---

## Categorical features

```text
Meal
Country
MarketSegment
DistributionChannel
ReservedRoomType
AssignedRoomType
DepositType
CustomerType
```

---

## Identifier-like categorical features

```text
Agent
Company
```

These require special consideration because they can have many categories.

---

# 46. Potential Feature Engineering

We should consider, but not blindly create, features such as:

### Total nights

```text
TotalNights =
    StaysInWeekendNights
    +
    StaysInWeekNights
```

### Total guests

```text
TotalGuests =
    Adults
    +
    Children
    +
    Babies
```

### Room changed

```text
RoomChanged =
    ReservedRoomType != AssignedRoomType
```

### Previous booking activity

Potentially:

```text
TotalPreviousBookings =
    PreviousCancellations
    +
    PreviousBookingsNotCanceled
```

### Cancellation history rate

Potentially:

```text
PreviousCancellationRate =
    PreviousCancellations /
    TotalPreviousBookings
```

But we need to handle zero denominators.

These are **feature engineering ideas**, not yet approved project decisions.

---

# 47. Potential Leakage Checklist

Before training either model, we should explicitly review:

```text
ReservationStatus
ReservationStatusDate
IsCanceled
ADR
```

depending on which target is being predicted.

For example:

## Classification

```text
Target = IsCanceled
```

Do not let the model see information that is only known after cancellation.

---

## Regression

```text
Target = ADR
```

ADR itself obviously cannot be included as a feature.

Also, features that were determined after the final price/outcome may need consideration.

---

# 48. What We Should NOT Do Yet

Do not immediately:

```text
❌ Delete all duplicates
❌ Delete all ADR = 0 rows
❌ Delete ADR = 5400
❌ Delete all zero-adult bookings
❌ Delete all zero-night bookings
❌ Fill every missing value with 0
❌ Encode every categorical value as an integer
❌ Scale everything immediately
❌ Use all 31 columns
```

Every transformation should have a reason.

---

# 49. What We Know About Dataset Quality

Overall:

### Structurally

The dataset looks good.

```text
79,330 rows
31 columns
```

### Missing data

Very small amount:

```text
28 rows
```

### Numerical validity

Good:

```text
No negative count values
No invalid target values
```

### Date validity

Good:

```text
No invalid calendar dates
```

### Target consistency

Good:

```text
ReservationStatus ↔ IsCanceled
```

### Main concerns

The major issues are:

```text
1. Many exact duplicates
2. ADR = 0 records
3. Extreme ADR outliers
4. Zero-adult bookings
5. Zero-night bookings
6. Potential target leakage
7. Small amount of missing data
```

---

# 50. Current Data Understanding Status

## Completed

- [x] Dataset dimensions
- [x] Column names
- [x] Data types
- [x] Missing-value investigation
- [x] Duplicate investigation
- [x] Target validity
- [x] ADR investigation
- [x] Non-negative count checks
- [x] Arrival component validity
- [x] Full calendar date validity
- [x] Cancellation-status consistency
- [x] Booking logical consistency

These checks are implemented in the validation script.

---

# 51. What Comes Next?

The logical next stage is **EDA**.

We should now investigate the dataset statistically and visually.

## EDA should answer questions such as:

### Dataset distribution

```text
How are numerical variables distributed?
```

### Classification

```text
What is the cancellation rate?
Which features appear related to cancellation?
```

### Regression

```text
What does the ADR distribution look like?
How severe is its skew?
```

### Time

```text
How does cancellation vary by:
- year?
- month?
```

### Booking behavior

```text
Does LeadTime affect cancellation?
Does stay length affect ADR?
```

### Customer behavior

```text
Does repeated-guest status affect cancellation?
Does previous cancellation history matter?
```

### Booking channels

```text
Which market segments have higher cancellation rates?
Which distribution channels have different behavior?
```

### Room information

```text
Does room type affect ADR?
Does a room assignment change relate to cancellation?
```

---

# 52. Recommended Project Flow From Here

```text
                    RAW DATA
                       │
                       ▼
                DATA VALIDATION
                       │
                       ▼
              DATA UNDERSTANDING
                       │
                       ▼
                     EDA
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   CLASSIFICATION              REGRESSION
   IsCanceled                     ADR
          │                         │
          └────────────┬────────────┘
                       ▼
               DATA CLEANING
                       │
                       ▼
              FEATURE ENGINEERING
                       │
                       ▼
                PREPROCESSING
                       │
                       ▼
                TRAIN / TEST SPLIT
                       │
                       ▼
                 MODEL TRAINING
                       │
                       ▼
                 MODEL EVALUATION
                       │
                       ▼
                 MODEL COMPARISON
                       │
                       ▼
                 FINAL RESULTS
```

---

# 53. Team Mental Model

If someone on the team asks:

> "What do we actually know about the database?"

The answer should be:

> We have a 79,330-row, 31-column hotel-booking dataset. Each row represents a booking record containing arrival information, stay duration, guest information, booking channel information, customer history, room information, deposit information, price information, and the final reservation outcome. We have validated the structure, targets, numerical constraints, dates, and cancellation consistency. The dataset has only 28 rows with missing values, but it contains 25,902 exact duplicate rows, 1,208 zero-ADR records, two ADR values above 500, and some bookings with zero adults or zero nights that require investigation. Before modeling, we need to complete EDA and make explicit decisions about duplicates, unusual records, missing values, feature engineering, preprocessing, and target leakage.

---

# 54. One-Sentence Meaning of Every Column

| Column                        | Meaning                            |
| ----------------------------- | ---------------------------------- |
| `IsCanceled`                  | Whether the booking was canceled   |
| `LeadTime`                    | Days between booking and arrival   |
| `ArrivalDateYear`             | Arrival year                       |
| `ArrivalDateMonth`            | Arrival month                      |
| `ArrivalDateWeekNumber`       | Arrival week number                |
| `ArrivalDateDayOfMonth`       | Arrival day                        |
| `StaysInWeekendNights`        | Weekend nights                     |
| `StaysInWeekNights`           | Weekday nights                     |
| `Adults`                      | Number of adults                   |
| `Children`                    | Number of children                 |
| `Babies`                      | Number of babies                   |
| `Meal`                        | Meal plan                          |
| `Country`                     | Customer country                   |
| `MarketSegment`               | Booking market segment             |
| `DistributionChannel`         | Booking distribution channel       |
| `IsRepeatedGuest`             | Whether customer is a repeat guest |
| `PreviousCancellations`       | Previous canceled bookings         |
| `PreviousBookingsNotCanceled` | Previous non-canceled bookings     |
| `ReservedRoomType`            | Originally reserved room type      |
| `AssignedRoomType`            | Assigned room type                 |
| `BookingChanges`              | Number of booking changes          |
| `DepositType`                 | Deposit type                       |
| `Agent`                       | Booking agent identifier           |
| `Company`                     | Company identifier                 |
| `DaysInWaitingList`           | Days spent on waiting list         |
| `CustomerType`                | Customer category                  |
| `ADR`                         | Average Daily Rate                 |
| `RequiredCarParkingSpaces`    | Requested parking spaces           |
| `TotalOfSpecialRequests`      | Number of special requests         |
| `ReservationStatus`           | Final reservation status           |
| `ReservationStatusDate`       | Date associated with final status  |

---

# 55. Final Checklist for the Team

Before we say:

> "The data is ready for ML"

we should be able to answer **YES** to all of these:

### Understanding

- [ ] Does everyone understand what one row represents?
- [ ] Does everyone understand every column?
- [ ] Do we know which columns are targets?
- [ ] Do we know which columns are potential leakage?

### Data quality

- [x] Missing values investigated
- [x] Duplicates investigated
- [x] Negative values checked
- [x] Target validity checked
- [x] Date validity checked
- [x] Cancellation consistency checked
- [x] Logical booking anomalies investigated

### EDA

- [ ] Numerical distributions
- [ ] Categorical distributions
- [ ] Target distributions
- [ ] Correlation analysis
- [ ] Cancellation analysis
- [ ] ADR analysis
- [ ] Time-based analysis
- [ ] Outlier analysis

### Cleaning

- [ ] Duplicate strategy decided
- [ ] Missing-value strategy decided
- [ ] ADR zero strategy decided
- [ ] ADR outlier strategy decided
- [ ] Unusual booking strategy decided

### ML preparation

- [ ] Features selected
- [ ] Leakage removed
- [ ] Categorical encoding decided
- [ ] Numerical preprocessing decided
- [ ] Feature engineering decided
- [ ] Train/test split designed
- [ ] Preprocessing fitted only on training data

---

# Bottom Line

The database is **large enough to support serious ML work**, and the initial validation shows that the basic structure is reasonably sound.

However, the most important lesson is:

> **We should not confuse "the code ran successfully" with "the data is ready for modeling."**

The validation has established that the data is structurally valid in many important ways, but we still need to understand its **statistical behavior and business meaning**.

The next major stage should therefore be:

```text
VALIDATION
    ↓
EDA
    ↓
DECISIONS ABOUT DATA QUALITY
    ↓
CLEANING
    ↓
FEATURE ENGINEERING
    ↓
PREPROCESSING
    ↓
MODELING
```

And the biggest things currently requiring team decisions are:

```text
1. What to do with the 25,902 exact duplicate rows
2. What zero ADR actually represents
3. How to handle the extreme ADR values
4. What the zero-adult / zero-night bookings represent
5. How to handle the 28 missing values
6. Which columns create target leakage
7. Which information would realistically be available when making each prediction
```