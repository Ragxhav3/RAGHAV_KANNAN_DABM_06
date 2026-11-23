
# DABM Experiment No. 6 – Market Basket Analysis (Association Rule Mining using Apriori Algorithm)

## Aim

To perform Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from transaction data for decision-making in retail and business analytics.

## Dataset

A sample retail transaction dataset containing **10 transactions** across **4 grocery items**: `Bread`, `Butter`, `Milk`, and `Eggs`.

| Transaction ID | Items Purchased |
|:---:|---|
| T1 | Milk, Bread, Eggs |
| T2 | Milk, Bread |
| T3 | Milk, Eggs |
| T4 | Bread, Butter |
| T5 | Milk, Bread, Butter, Eggs |
| T6 | Bread, Eggs |
| T7 | Milk, Butter |
| T8 | Milk, Bread, Butter |
| T9 | Bread, Butter, Eggs |
| T10 | Milk, Bread, Butter, Eggs |

*The transaction list is transformed into a One-Hot Encoded binary DataFrame prior to mining.*

## Algorithm / Steps Followed

1. Import required Python libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `mlxtend`).
2. Create the sample grocery transaction dataset.
3. Convert the dataset into a one-hot encoded binary matrix using `TransactionEncoder`.
4. Apply the `apriori()` algorithm (`min_support=0.3`) to identify frequent itemsets.
5. Generate association rules from frequent itemsets using `association_rules()` with `lift >= 1.0`.
6. Filter strong rules based on thresholds (`confidence >= 0.6` and `lift > 1`).
7. Interpret the rules to uncover cross-selling and store layout insights.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
|---|---|---|
| Data Handling | `TransactionEncoder()` | Converts transactions into a one-hot encoded binary matrix |
| Frequent Pattern Mining | `apriori()` | Finds frequent itemsets meeting the minimum support threshold |
| Association Rule Generation | `association_rules()` | Generates rules using support, confidence, and lift |
| Filtering | `rules[...]` | Selects rules meeting confidence and lift thresholds |
| Display | `print()` | Displays transaction data, itemsets, and rules |
| Display | `DataFrame.head()` | Displays the first rows of the DataFrame |

---

## Input 1

```python
# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules

# Step 2: Create Sample Transaction Dataset
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Butter'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

dataset
```

## Output 1

```text
[['Milk', 'Bread', 'Eggs'],
 ['Milk', 'Bread'],
 ['Milk', 'Eggs'],
 ['Bread', 'Butter'],
 ['Milk', 'Bread', 'Butter', 'Eggs'],
 ['Bread', 'Eggs'],
 ['Milk', 'Butter'],
 ['Milk', 'Bread', 'Butter'],
 ['Bread', 'Butter', 'Eggs'],
 ['Milk', 'Bread', 'Butter', 'Eggs']]
```

---

## Input 2

```python
# Step 3: Convert Dataset into One-Hot Encoding
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)

df = pd.DataFrame(te_array, columns=te.columns_)

print("----- Transaction Dataset (One-Hot Encoded) -----")
print(df.head())
```

## Output 2

```text
----- Transaction Dataset (One-Hot Encoded) -----
   Bread  Butter   Eggs   Milk
0   True   False   True   True
1   True   False  False   True
2  False   False   True   True
3   True    True  False  False
4   True    True   True   True
```

---

## Input 3

```python
# Step 4: Apply Apriori Algorithm
frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\n----- Frequent Itemsets -----")
print(frequent_itemsets)
```

## Output 3

```text
----- Frequent Itemsets -----
    support               itemsets
0       0.8                (Bread)
1       0.6               (Butter)
2       0.6                 (Eggs)
3       0.7                 (Milk)
4       0.5        (Butter, Bread)
5       0.5          (Eggs, Bread)
6       0.5          (Milk, Bread)
7       0.3         (Butter, Eggs)
8       0.4         (Butter, Milk)
9       0.4           (Milk, Eggs)
10      0.3  (Butter, Eggs, Bread)
11      0.3  (Butter, Milk, Bread)
12      0.3    (Eggs, Milk, Bread)
```

### Interpretation of Frequent Itemsets

- Bread has the highest support of 0.8 (80%).
- Milk has a support of 0.7 (70%).
- Butter and Eggs each have a support of 0.6 (60%).
- Bread and Butter occur together in 50% of transactions.
- Bread and Eggs occur together in 50% of transactions.

---

## Input 4

```python
# Step 5: Generate Association Rules
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

print("\n----- Association Rules -----")
print(rules[['antecedents', 'consequents',
             'support', 'confidence', 'lift']])
```

## Output 4

```text
----- Association Rules -----
        antecedents      consequents  support  confidence      lift
0          (Butter)          (Bread)      0.5    0.833333  1.041667
1           (Bread)         (Butter)      0.5    0.625000  1.041667
2            (Eggs)          (Bread)      0.5    0.833333  1.041667
3           (Bread)           (Eggs)      0.5    0.625000  1.041667
4    (Butter, Eggs)          (Bread)      0.3    1.000000  1.250000
5   (Butter, Bread)           (Eggs)      0.3    0.600000  1.000000
6     (Bread, Eggs)         (Butter)      0.3    0.600000  1.000000
7          (Butter)    (Bread, Eggs)      0.3    0.500000  1.000000
8            (Eggs)  (Butter, Bread)      0.3    0.500000  1.000000
9           (Bread)   (Butter, Eggs)      0.3    0.375000  1.250000
10    (Milk, Bread)         (Butter)      0.3    0.600000  1.000000
11         (Butter)    (Milk, Bread)      0.3    0.500000  1.000000
12    (Milk, Bread)           (Eggs)      0.3    0.600000  1.000000
13           (Eggs)    (Milk, Bread)      0.3    0.500000  1.000000
```

### Interpretation of Association Rules

Association rules identify relationships between products purchased together.

- **Support:** Indicates how frequently an item combination occurs in all transactions.
- **Confidence:** Indicates how frequently the consequent item is purchased when the antecedent item is purchased.
- **Lift:** Measures the strength of the association compared with independent purchasing.

---

## Input 5

```python
# Step 6: Sort and Filter Strong Rules
strong_rules = rules[
    (rules['confidence'] >= 0.6) &
    (rules['lift'] > 1)
]

print("\n----- Strong Association Rules -----")
print(strong_rules[['antecedents', 'consequents',
                    'support', 'confidence', 'lift']])
```

## Output 5

```text
----- Strong Association Rules -----
      antecedents consequents  support  confidence      lift
0        (Butter)     (Bread)      0.5    0.833333  1.041667
1         (Bread)    (Butter)      0.5    0.625000  1.041667
2          (Eggs)     (Bread)      0.5    0.833333  1.041667
3         (Bread)      (Eggs)      0.5    0.625000  1.041667
4  (Butter, Eggs)     (Bread)      0.3    1.000000  1.250000
```

### Strong Association Rules Interpretation

The following relationships were identified:

1. Customers purchasing Butter also purchase Bread with 83.33% confidence.
2. Customers purchasing Bread also purchase Butter with 62.5% confidence.
3. Customers purchasing Eggs also purchase Bread with 83.33% confidence.
4. Customers purchasing Bread also purchase Eggs with 62.5% confidence.
5. Customers purchasing Butter and Eggs together also purchase Bread with 100% confidence.

---

## Result

The Apriori algorithm successfully discovered frequent itemsets and strong association rules from the retail transaction dataset.

The analysis identified that Bread, Butter, and Eggs frequently appear together in customer transactions. The strongest identified rule was that customers purchasing Butter and Eggs also purchased Bread, with 100% confidence and a lift value of 1.25.

Thus, the experiment demonstrates how Market Basket Analysis can identify customer purchasing patterns and support cross-selling strategies, product placement, inventory planning, and store layout optimization.
