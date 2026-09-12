# Task 6 — K-Nearest Neighbors (KNN) Classification
**Elevate Labs — AI & ML Internship**

## Objective
Understand and implement KNN for classification problems.

## Tools Used
- Python
- Scikit-learn
- Pandas
- Matplotlib / Seaborn

## Dataset
Iris dataset (built into `sklearn.datasets.load_iris`) — 150 samples, 4 numeric features (sepal length/width, petal length/width), 3 balanced classes: setosa, versicolor, virginica (50 each). No missing values. Also exported to `data/iris.csv` for reference.

## Project Structure
```
iris-knn/
├── data/
│   └── iris.csv
├── plots/
│   ├── 01_accuracy_vs_k.png
│   ├── 02_confusion_matrix.png
│   ├── 03_decision_boundary.png
│   └── 04_decision_boundary_by_k.png
├── knn.py
└── README.md
```

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python knn.py
```
Prints per-K accuracy and the final evaluation to the console, and saves all charts to `plots/`.

## Approach
1. **Normalize features**: split into train/test (80/20, stratified), then standardized all features with `StandardScaler` — essential for KNN since it's a distance-based algorithm and unscaled features (e.g., petal length in cm vs. a 0-1 ratio) would distort distance calculations.
2. **Experiment with K**: trained `KNeighborsClassifier` for every K from 1 to 25 and tracked test accuracy for each.
3. **Select K**: chose a K that gives strong, stable accuracy without going as low as K=1 (which is more sensitive to noise).
4. **Evaluate**: accuracy, classification report, and confusion matrix on the test set.
5. **Visualize decision boundaries**: trained a 2D KNN model on just `petal length` and `petal width` (the two most separable features) to plot the actual decision regions, and compared boundary smoothness at K=1, the selected K, and K=25.

## Results

**K selection**
- Accuracy fluctuates between 90–97% across K=1 to 25, peaking at 96.7% for several K values (including K=1, K=7, and K=9 through K=21).
- Since K=1 fits training noise most closely (each point is classified by its single nearest neighbor), a mid-range K was chosen instead: **K = 14**, which sits comfortably inside the stable high-accuracy plateau.

**Final model (K=14) evaluation**
- Test Accuracy: **96.7%** (29/30 correct)
- Setosa: perfect precision and recall (linearly separable from the other two species).
- Versicolor and virginica have one point misclassified between them — these two species overlap slightly in feature space, which is a well-known characteristic of this dataset.

**Decision boundary visualization**
- At **K=1**, the boundary is jagged and tightly wraps around individual points — overfitting to local noise.
- At **K=14** (selected), the boundary is much smoother while still correctly separating the three species.
- At **K=25**, the boundary is very smooth but starts to lose some of the finer separation between versicolor and virginica — too much smoothing (underfitting) as K grows very large relative to the dataset size.

This illustrates the classic KNN trade-off: small K → low bias/high variance (overfits to noise); large K → high bias/low variance (oversmooths class boundaries).

---
*Submitted as part of the Elevate Labs AI & ML Internship (MSME, Govt. of India).*
