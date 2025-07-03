# 🎬 Movie Recommendation System

A personalized movie recommendation engine that leverages machine learning techniques to provide user-specific content suggestions. This system is built using collaborative filtering techniques including Neural Collaborative Filtering (NCF) and Matrix Factorization (MF).

---

## 📌 Features

- 📊 Collaborative filtering using user-item interactions
- 🧠 Neural networks applied to recommendation logic
- 📁 Preprocessing of raw movie ratings data
- 🔍 Evaluation with RMSE and top-K precision/recall
- 💬 Clean, well-commented Jupyter notebooks for reproducibility

---

## 🛠 Technologies Used

- **Languages:** Python
- **Libraries:** Pandas, NumPy, TensorFlow, Scikit-learn, Matplotlib
- **Models:** Neural Collaborative Filtering (NCF), Matrix Factorization (MF)
- **Data Source:** [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

---

## 🧪 How It Works

1. **Data Preparation:**
   - Load and clean MovieLens data
   - Encode user and movie IDs
   - Split into train/test sets

2. **Model Training:**
   - Build and compile NCF/MF models
   - Train using sparse matrices

3. **Evaluation:**
   - Generate predictions
   - Calculate RMSE, precision@k, recall@k

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
