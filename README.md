# Customer Feedback Analysis and Automated Response

This project analyzes Amazon product review data (`feedbackdata.csv`) to uncover customer satisfaction trends, identify the most critical negative reviews, and automatically draft personalized apology/resolution emails for the top complaints using a generative AI model (Google Gemini).

The full workflow lives in **`reviewReport.ipynb`**.

---

## Tech Stack

- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly.express`
- **Text processing:** `re`, `collections.Counter`
- **Automation / GenAI:** `google-genai` (Gemini API)
- **Config / secrets:** `python-dotenv` (loads settings from a `.env` file)

---

## Workflow

### 1. Imports

Loads all libraries needed for data wrangling, plotting, and NLP, and suppresses non-critical warnings for a cleaner notebook output. Also installs and imports `python-dotenv`, which is used to load configuration (file paths, API keys) from a local `.env` file instead of hardcoding them in the notebook.

### 2. Loading Data

Calls `load_dotenv()` to read the `.env` file, then reads the raw review dataset into a pandas DataFrame from the path specified by the `FILE_PATH` environment variable (rather than a hardcoded path to `feedbackdata.csv`).

### 3. Data Cleaning & Preprocessing

> **Design decision (noted in-notebook):** rows are intentionally **not dropped**, so that potentially critical negative-review data isn't lost during cleaning.

Steps taken:

- **Explore structure:** `describe()`, `dtypes`, `info()` to understand columns, types, and null counts.
- **Drop low-value columns:** removed `reviews.didPurchase`, `reviews.id`, `reviews.userCity`, `reviews.userProvince` — high null counts / low relevance to the analysis.
- **Check duplicates:** verified duplicate row count.
- **Clean product names:** removed repeated name fragments (e.g., `Echo(White),,,Echo(White)` → `Echo(White)`) and trailing commas/spaces.
- **Fill missing product names:** used a custom `asins` → `name` lookup table to manually backfill missing product names where the ASIN was known.
- **Derive review source:** parsed `reviews.sourceURLs` with `urlparse` to extract the domain (e.g., `amazon`, `bestbuy`, `walmart`) into a new `source` column, stripping `www.` and `.com`.
- **Drop redundant columns:** removed `keys`, `reviews.sourceURLs`, `asins` based on domain knowledge/correlation — no longer needed after `source` was derived.
- **Impute missing ratings using sentiment keywords:** scanned `reviews.title` + `reviews.text` for positive keywords (e.g., "excellent", "highly recommend") to fill missing ratings as **4**, and negative keywords (e.g., "defective", "worst", "broken") to fill as **2**. Ratings still missing after this pass were filled as a **neutral 3**, to avoid introducing rating bias (i.e., never auto-assigning the extreme 1 or 5 values).
- **Standardize dates:** converted `reviews.date`, `reviews.dateAdded`, and `reviews.dateSeen` to proper date objects (the latter parsed from the last value in a comma-separated list of timestamps).

### 4. Exploratory Data Analysis (EDA)

**KPI Dashboard**
Calculates and renders (as styled HTML cards) three headline metrics:

- Total Reviews
- Average Rating
- Recommendation Rate (% of reviewers who marked `doRecommend = True`)

**Product Analysis**

- **Top 10 highest-rated products** (min. 50 reviews) — bar chart of average rating.
  - _Insight:_ No strong link between review volume and rating; accessories (e.g., the Kindle Fire HD 9w Charger) underperform core devices; flagship Fire tablets/Echo cluster around 4.4–4.6.
- **Lowest-rated products** (min. 50 reviews) — table view.

**Rating Analysis**

- **Overall rating distribution** — pie chart.
  - _Insight:_ Heavily skewed positive (68.6% 5-star, 24.7% 4-star ≈ 93% positive); only ~1.2% each for 1- and 2-star, consistent with typical e-commerce self-selection bias.
- **Source-wise rating distribution** (log scale) and **source-wise rating % heatmap**.
  - _Insight:_ Amazon and BestBuy dominate review volume, making their distributions reliable; small sources (Walmart, Sears, eBay) show extreme percentages (e.g., Walmart at 100% 5-star) that are artifacts of tiny sample sizes, not genuine sentiment.

**Helpful Votes Analysis**

- **Top 10 most-helpful reviews** — table.
  - _Insight:_ Echo, Fire Tablet, and Fire TV dominate; even a 1-star review made the top-10 helpful list, showing critical reviews carry real influence.
- **Average helpfulness by rating** — bar chart.
  - _Insight:_ Strong inverse relationship — 1-star reviews get ~8 average helpful votes vs. ~0.5 for 5-star, meaning shoppers lean on negative reviews far more when deciding to buy.
- **Focus products** (avg rating ≤ 3, sorted by max helpful votes) — flags specific products where low ratings are being actively validated/upvoted by other shoppers, signaling a real (not isolated) product problem.

**Time Analysis**

- **Monthly review trend** — line chart of review volume over time, resampled to monthly buckets.

**Negative Reviews Analysis**

- Filters to `reviews.rating < 3` to isolate negative reviews.
- Extracts a curated list of complaint keywords (e.g., "defective", "broken", "garbage", "flimsy", "overheats") and counts their frequency across negative review text using regex + `Counter`.
- Visualizes the result as a **treemap ("Word Frequency")**.
  - _Insight:_ "Defective" and "worst" dominate (~65% of flagged terms combined), pointing to core product-quality/hardware failure as the primary driver of dissatisfaction, with delivery ("late") as a secondary, distinct issue.

### 5. Automation — Prioritizing & Responding to Critical Reviews

**Step 1 — Score and rank negative reviews**
Computes a `priority_score` for each negative review:

```
priority_score = numHelpful × rating_weight × review_text_length
```

where `rating_weight` gives more weight to 1-star (5x) than 2-star (4x) reviews. This surfaces reviews that are both highly negative, well-supported by other shoppers (helpful votes), and detailed (longer text = more substantiated complaint). The top 10 by this score are shortlisted.

**Step 2 — Filter to the most actionable complaints**
From that top 10, a second keyword filter (e.g., "broken", "refund", "defective", "danger", "cracked", "stopped working", "replacement") narrows the list down to the **top 3 most critical, actionable reviews** (`critical3`).

**Step 3 — Generate personalized responses with Gemini**
For each of the 3 critical reviews, a structured prompt is sent to Google's `gemini-3.1-flash-lite` model instructing it to:

- Address the customer as "Dear Customer,"
- Acknowledge the specific issue(s) raised
- Apologize sincerely and explain the issue matters to the company
- Offer a resolution (refund, replacement, troubleshooting, or support)
- Stay within 120–180 words
- Avoid inventing details not present in the review

The Gemini client is initialized using the `GEMINI_API_KEY` value loaded from the `.env` file via `os.getenv("GEMINI_API_KEY")`.

The three generated emails are stored in the `emails` list and also included in the notebook as a pre-generated reference ("Pregenerated Emails" section) covering Fire TV hardware defects, connectivity/UX issues, and unresponsive-remote/performance complaints.

---

## Key Insights Summary

| Analysis               | Insight                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Product ratings        | No correlation between review volume and rating; accessories underperform core devices                                                      |
| Rating distribution    | ~93% of reviews are positive (4–5 stars); negative feedback is rare but disproportionately "helpful"                                        |
| Source reliability     | Amazon/BestBuy provide statistically reliable distributions; small sources are noisy                                                        |
| Helpfulness            | 1-star reviews are ~16x more "helpful" on average than 5-star reviews                                                                       |
| Negative review themes | Complaints center on hardware defects (defective, broken, flimsy, overheats), not usability or price                                        |
| Automation             | The pipeline reliably surfaces the most damaging, well-substantiated complaints and drafts on-brand, non-fabricated apology emails for them |

---

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly google-genai python-dotenv
   ```
2. Create a `.env` file in the project root (see `.env.example` below, or create your own) with the following variables:
   ```env
   FILE_PATH=/path/to/your/feedbackdata.csv
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

   - `FILE_PATH` — path to your local copy of the review dataset (`feedbackdata.csv`). This is read in the **Loading Data** cell via `load_dotenv()` + `os.getenv("FILE_PATH")`, so no code changes are needed to point the notebook at your data.
   - `GEMINI_API_KEY` — your Google Gemini API key, used to authenticate the `google-genai` client in the Automation section.
3. Run all cells top to bottom. `load_dotenv()` is called early in the notebook, so both variables are available by the time they're needed — the Automation section will pick up `GEMINI_API_KEY` automatically without any further setup.

> **Note:** Keep your `.env` file out of version control (add it to `.gitignore`) since it contains your API key.

---
