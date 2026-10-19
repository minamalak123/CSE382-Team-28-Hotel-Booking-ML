# CSE382 Hotel Booking ML Project (Team 28)

End-to-end machine learning project on the City Hotel (H2) booking dataset.
Tasks: Regression (ADR), Classification (IsCanceled), Clustering.

## Team Members

| Name | Student ID | GitHub Username |
|------|------------|-----------------|
| Mina Malak Mounir Abdelmalek | 2400726 | minamalak123 |
| Abdelrahman Amro Hussein Mansour | 24P0350 | Abdelrahman0350 |
| Omar Ramy Mohamed Ahmed Salama | 24P0252 | OmarRamy997 |
| Aly Ahmed Mohamed Elsayed Eltallawy | 24P0279 | Aly-Tallawy |
| Ahmed Baher Amir Mahmoud El Damhougy | 24P0274 | ahmedbaher905-png |

## Repository

- GitHub: https://github.com/minamalak123/CSE382-Team-28-Hotel-Booking-ML
- Submitted commit SHA: (fill in right before submission)

## Environment Setup (Windows)

1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Clone the repository and open a terminal in the project folder.
3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate it:

```bash
.venv\Scripts\activate
```

5. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The raw data file is NOT stored in Git. To get it:

1. Download H2.csv from:
   https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv
2. Save it as `data/raw/H2.csv`

## How to Run

Run the initial data validation script:

```bash
python src/data/validate_data.py
```

## Project Status

Milestone 1 (in progress): data validation, EDA, preprocessing, baseline experiments.