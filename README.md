# 🏦 Bank Marketing Analysis

> Comprehensive analysis of a Portuguese bank marketing dataset using **R** and **Python**, applying multiple machine learning algorithms to predict term deposit subscriptions.  
> This project demonstrates advanced data mining, model comparison, feature importance analysis, and clustering for actionable business insights.

---

## 📊 Project Overview
This project explores the **Bank Marketing Dataset** consisting of **41,188 rows and 21 columns**, with features describing client demographics, campaign interactions, and economic indicators.

The primary goal is to predict whether a customer will subscribe to a **term deposit (subscription)** after a marketing campaign.  
We implement **supervised** and **unsupervised learning models** to evaluate predictive performance and uncover hidden patterns.

---

## 🎯 Objectives
- Predict **term deposit subscriptions** using classification algorithms.  
- Handle **imbalanced data** through proper training–testing splits.  
- Compare the performance of different models across R and Python.  
- Determine the most influential variables using **feature importance** techniques.  
- Segment customers via clustering for targeted marketing insights.

---

## 🧠 Techniques & Algorithms
### 🔹 Supervised Learning (Categorical Data)
- Decision Tree 🌳  
- Naive Bayes 📈  
- Random Forest 🌲  
- Gradient Boosting 🚀  
- Bagging 🎲  
- Support Vector Machine (SVM) ⚔  

### 🔹 Supervised Learning (Numerical Data)
- K-Nearest Neighbors (KNN) 👥  
- Decision Tree 🌳  
- Naive Bayes 📈  

### 🔹 Unsupervised Learning
- K-Means Clustering  
- K-Medoids (PAM)  
- DBSCAN (Density-Based Spatial Clustering)  
- PCA (Principal Component Analysis)  

---

## 🧩 Key Features
- End-to-end model implementation in both **R** and **Python**.  
- Uses the `caret` package for cross-validation, training, and accuracy metrics.  
- Handles categorical and numerical features separately.  
- Evaluates **confusion matrices**, **heatmaps**, and **feature importance** for clarity.  
- Performs **dimensionality reduction (PCA)** to improve interpretability.  
- Conducts **customer segmentation** through clustering algorithms.  

---

## ⚙ Tech Stack
| Category | Tools |
|-----------|-------|
| Language | R, Python |
| Data Processing | dplyr, pandas |
| Machine Learning | caret, rpart, randomForest, gbm, xgboost, e1071 |
| Visualization | Matplotlib, Seaborn, R heatmaps |
| Output Files | `Bank.R`, `Bank.ppt` |

---

## 🧪 Implementation Highlights
- Data split using `createDataPartition` (70-30 train-test ratio).  
- Applied multiple classification models using `train()` from **caret**.  
- Visualized confusion matrices as heatmaps for each algorithm.  
- Used **Random Forest** and **Boosting** for feature importance ranking.  
- Employed **PCA** to identify the optimal number of components (4).  
- Clustered customers using **K-Means**, **K-Medoids**, and **DBSCAN**.

---

## 📈 Insights Derived
- **Decision Tree** and **Random Forest** offered balanced performance and interpretability.  
- **Gradient Boosting** achieved the **highest predictive accuracy** among all models.  
- **PCA** revealed 4 key principal components contributing most variance.  
- **Clustering** identified distinct customer segments, aiding targeted campaign design.  
- **Economic indicators** and **contact duration** emerged as the strongest predictors of subscription likelihood.  
- The model outputs can support **personalized marketing**, **conversion optimization**, and **customer retention** strategies.

---

## 🧠 Feature Importance
- Random Forest ranked **duration**, **poutcome**, **age**, and **contact month** as the most influential features.  
- Boosting models confirmed the **contact type** and **housing loan** status also contribute significantly to prediction accuracy.  

---

## 🚀 Future Enhancements

-🧩 Integrate cross-language pipelines: Combine R and Python models in a unified workflow for comparison automation.
-🧠 Hyperparameter tuning: Apply grid search and Bayesian optimization for boosting models.
-🌐 Interactive dashboards: Use Power BI or R Shiny to visualize campaign metrics dynamically.
-⚙ MLOps integration: Deploy the best model through CI/CD pipelines for real-time predictions.
-🌎 Dataset expansion: Include multi-country bank datasets to enhance generalization.
-🔍 Explainable AI (XAI): Integrate SHAP or LIME for transparency in model decisions.

---

## 📂 Repository Structure

-📦 Bank-Marketing-Analysis
-┣ 📜 Bank.R → R implementation of all ML models
-┣ 🖼 Bank.ppt → Presentation summarizing findings and visuals
-┣ 📄 README.md → Project overview and documentation

---

## 👨‍💻 About the Developers

*Krishna Kanth Reddy K*  
🎓 MPS in Analytics, Northeastern University, Vancouver  
💼 Data Analyst (4+ years in SQL, ETL, Power BI, and Python)  
📧 [krishnakanthreddycan@gmail.com](mailto:krishnakanthreddycan@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/krishnakrk)  

*Aquila Pillay*  
🎓 MPS in Analytics, Northeastern University, Vancouver  
💼 Data Enthusiast with experience in analytics, visualization, and storytelling (1+ years in SQL, ETL, Power BI, and Python) 
📧 [aquilapersis@gmail.com](mailto:aquilapersis@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/aquilapillay)

---
