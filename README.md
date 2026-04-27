# S-P_Challange

This repository includes two prediction scripts that build and evaluate a machine learning model for forecasting stock returns using price and news data.

- `predict.py`: uses the `sentence-transformers` library to convert news headlines and summaries into embeddings, reduces their dimensionality with PCA, merges them with stock price features, trains an XGBoost regression model, and evaluates both regression and directional trading signal performance.
- `predict_ollama_embeddings.py`: follows a similar workflow but obtains news embeddings from a local Ollama-like embedding API instead of using `sentence-transformers` directly.

Both scripts compute technical indicators, target next-day returns, train a time-series-aware model, and visualize model performance, classification metrics, and strategy returns.