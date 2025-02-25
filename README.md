# ML-Powered Data Studio

## Overview
ML-Powered Data Studio is a Python-based desktop application designed to streamline data analysis, visualization, and transformation. Built with `ttkbootstrap` for a modern UI and `Plotly` for interactive charts, this tool allows users to clean, preprocess, and forecast data without extensive coding.

## Features

### 1. File & Data Management
- **File Upload:** Supports CSV and Excel file uploads for easy data import.
- **Column Selection & Chart Suggestions:** Users can select columns for the X-axis, Y-axis, and optionally the Z-axis (for 3D charts). The app suggests relevant visualizations based on selected columns.
- **Custom Chart Creation:** Generate scatter, line, bar, area, bubble, pie, and other charts using `Plotly`, with customizable titles and colors.

### 2. Custom Dashboard
- **Multi-Chart Dashboard:** Users can configure multiple charts and integrate them into a single dashboard using `Plotly` subplots for better data insights.

### 3. File Conversion
- **Convert Data Formats:** Convert datasets into CSV, Excel, SQLite, JSON, Parquet, and XML formats, making data accessible for different applications.

### 4. Forecasting
- **Model Selection:** Choose from Linear, Polynomial, and ARIMA forecasting models.
- **Forecast Horizon & Confidence Intervals:** Define future time steps and visualize confidence intervals (for ARIMA models).
- **Scikit-Learn & Statsmodels Integration:** Utilizes `scikit-learn` for regression models and `statsmodels` for ARIMA forecasting.

### 5. Data Cleaning & Transformation
- **Cleaning Operations:** Remove duplicates, handle missing data (drop or fill values), standardize numeric data, fix structural errors, and manage outliers.
- **Data Transformation:** Apply log transformation and scaling; reset transformations when needed.

### 6. Data Integration
- **SQLite Integration:** Load data from SQLite databases by selecting a table.
- **API Integration:** Fetch JSON data from APIs and convert it into a DataFrame for analysis.

### 7. Data Summary & Filtering
- **Summary Statistics:** View dataset shape, missing values per column, and descriptive statistics.
- **Data Filtering:** Apply custom query filters to refine datasets and reset them to the original state when needed.

### 8. Advanced Tools
- **One-Hot Encoding & Feature Engineering:** Transform categorical data for machine learning models.
- **Integrated Data Processing:** Apply filters, review summaries, and preprocess data efficiently.

### 9. Settings
- **UI Customization:** Modify fonts, colors, and themes for a personalized experience.
- **Reset & Apply Configurations:** Restore default settings or apply changes instantly.

## Installation & Usage
### Prerequisites
- Python 3.7+
- Required libraries: Install via pip using the command:
  ```sh
  pip install pandas plotly sklearn ttkbootstrap statsmodels numpy
  ```

### Running the Application
```sh
python ml_data_studio.py
```

##       
##
