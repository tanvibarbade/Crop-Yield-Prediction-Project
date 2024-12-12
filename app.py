import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
import shap
import pickle
import joblib

# Load the dataset for EDA (example dataset, replace 'crop' with your actual dataset if needed)
data = pd.read_csv('agricultural_yield_train.csv')  # Replace with your actual dataset path

# Load the model
# model = joblib.load('random_forest_pickel.pkl')

# Set up the page configuration
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

# Title for the app
st.title('# 🌿💡 Agri-Analytics: Forecasting Yields from Soil to Sun 🌞💧')

# Sidebar navigation
page = st.sidebar.selectbox('Select Option', ['Home', 'EDA', 'Hypothesis Testing', 'Model Evaluation','Model'])

# Home Page
if page == 'Home':
    st.markdown("""
    ## Problem Statement  
    The problem at hand is predicting the **crop yield (kg per hectare)** based on various factors, such as:  
    - 🌱 **Soil quality**  
    - 🌾 **Seed variety**  
    - 🧪 **Fertilizer usage**  
    - 🌞 **Weather conditions** (sunny days and rainfall)  
    - 💧 **Irrigation practices**  

    The goal is to understand how each feature influences the crop yield and make accurate predictions based on these features.

    ## Objective of the Model  

    - 🎯 The primary objective is to predict the **crop yield (kg per hectare)** based on input features.  
    - 🔧 By analyzing the impact of factors like soil quality, fertilizer usage, and irrigation schedules, the model aims to help **optimize farming practices** for better yields.  
    - 🧠 Provide insights to farmers and agricultural planners on how to adjust factors such as fertilizer amounts or irrigation schedules based on **expected weather conditions and soil quality**.  
    - 📊 Help farmers plan their resources effectively by predicting yields under varying conditions.
    """)

# EDA Page
elif page == 'EDA':
    # Dataset Description
    st.markdown("""
    ## Dataset Columns Description 🌾📋  

    | **Column Name**                          | **Description**                                                                                     |  
    |------------------------------------------|-----------------------------------------------------------------------------------------------------|  
    | 🌱 **Soil_Quality**                       | Represents the quality of soil, measured on a scale (e.g., nutrient content, pH levels).             |  
    | 🌾 **Seed_Variety**                       | Indicates the type of seed used (e.g., 1 for hybrid, 0 for non-hybrid).                              |  
    | 🧪 **Fertilizer_Amount_kg_per_hectare**  | Quantity of fertilizer applied per hectare (in kilograms).                                          |  
    | 🌞 **Sunny_Days**                         | Total number of sunny days during the crop-growing season.                                          |  
    | 🌧️ **Rainfall_mm**                       | Amount of rainfall received (in millimeters) during the crop-growing season.                        |  
    | 💧 **Irrigation_Schedule**                | Represents the irrigation frequency/schedule (e.g., number of irrigations during the growing period).|  
    | 🎯 **Yield_kg_per_hectare**               | The target variable: Crop yield per hectare (in kilograms).                                         |  
    """)

    st.markdown("### Exploratory Data Analysis (EDA) 📊🔍📊")

    # Show the first few rows of the dataset
    st.markdown("<h3>First Few Rows of the Dataset:</h3>", unsafe_allow_html=True)
    st.write(data.head())

    # Shape of the dataset
    st.markdown("<h3>Rows and Columns:</h3>", unsafe_allow_html=True)
    Rows, Columns = data.shape
    st.write(f'Rows: {Rows} \nColumns: {Columns}')

    # Check for missing values and duplicates
    st.write("Missing values:")
    st.write(data.isnull().sum())

    st.write("Duplicates:")
    st.write(data.duplicated().sum())

    # Correlation of features
    st.write("Correlation of features:")
    st.write(data.corr())

    # Summary statistics
    st.write("Summary Statistics:")
    st.write(data.describe(include='all').fillna('-'))

    # Visualizations
    st.markdown("### 📊 Univariate Distribution of Agricultural Features:")

    # Set a dark theme with dark background and adjust color palette
    sns.set_theme(style="dark", palette="dark")

    # Create a list of features to visualize
    features = [
        "Soil_Quality", 
        "Seed_Variety", 
        "Fertilizer_Amount_kg_per_hectare", 
        "Sunny_Days", 
        "Rainfall_mm", 
        "Irrigation_Schedule", 
        "Yield_kg_per_hectare"
    ]

    # Create subplots for univariate analysis
    fig, axes = plt.subplots(4, 2, figsize=(8, 10))
    for i, feature in enumerate(features, 1):
        plt.subplot(4, 2, i)
        sns.histplot(data[feature], kde=True, color=sns.color_palette("Set2")[i % len(features)])
        plt.title(f"{feature}", fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(False)  # Disable the grid

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    - **Interpretation:**
        - **Soil Quality:** Even distribution of soil quality 🌱.
        - **Seed Variety:** Binary distribution, indicating two distinct seed varieties 🌾🌾.
        - **Fertilizer Amount:** Uniform spread of fertilizer 💧.
        - **Sunny Days:** Normally distributed with a peak around 100 sunny days ☀️.
        - **Rainfall:** Normally distributed with a peak around 500 mm 🌧️.
        - **Irrigation Schedule:** Multiple peaks, indicating variable irrigation practices 💦.
        - **Yield:** Normally distributed with a peak around 800 kg/ha 🌾.
    """)

    # Box Plot
    st.markdown("### 📊 Box Plot for Distribution of Data (Outliers Included): 📉")
    fig = plt.figure(figsize=(15, 4))
    sns.boxplot(data=data, palette="Set2", fliersize=5)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    - **Interpretation:**
      - **Soil Quality:** Low variability 🌱
      - **Seed Variety:** No variation 🌾
      - **Fertilizer Amount (kg/ha):** Significant variability 💩
      - **Sunny Days:** Consistent 🌞
      - **Rainfall (mm):** High variability 🌧️
      - **Irrigation Schedule:** No variation 💧
      - **Yield (kg/ha):** Significant variability 🌾📊
    """)

    # Violin Plot for Yield by Seed Variety
    st.markdown("### 📊 Violin Plot for Yield by Seed Variety: 🌱")
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(x='Seed_Variety', y='Yield_kg_per_hectare', data=data, palette='Set2')
    plt.title('Violin Plot: Yield by Seed Variety')
    plt.xlabel('Seed Variety')
    plt.ylabel('Yield (kg/ha)')
    st.pyplot(fig)

    st.markdown("""
    - **Interpretation** 🌱📊
        - The yield distribution for Seed Variety 1 🌾 is wider with a higher median compared to Seed Variety 0 🌾.
    """)

    # Violin Plot for Yield Distribution by Irrigation Schedule
    st.markdown("### 📊 Violin Plot for Yield Distribution by Irrigation Schedule: 💧")
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x='Irrigation_Schedule', y='Yield_kg_per_hectare', palette='muted', inner='quart', scale='width')
    st.pyplot(fig)

    st.markdown("""
    - **Interpretation** 🌾
    - The violin plot shows increasing crop yields 📈 with more irrigation events 💧, highlighting variability in distribution.
    - It underscores irrigation's significant impact on yields, providing insights for optimizing water usage and agricultural productivity 💡🌱.
    """)

    # Scatter Plot for Fertilizer Amount vs. Yield
    st.markdown("### 📊 Scatter Plot for Fertilizer Amount vs. Yield: 🌱")
    fig = plt.figure(figsize=(15, 5))
    sns.scatterplot(x='Fertilizer_Amount_kg_per_hectare', y='Yield_kg_per_hectare', data=data, color='orange')
    sns.regplot(x='Fertilizer_Amount_kg_per_hectare', y='Yield_kg_per_hectare', data=data, scatter=False, color='darkred', line_kws={"linewidth": 2, "color": "darkred"})
    plt.title('Scatter Plot: Fertilizer Amount vs. Yield')
    plt.xlabel('Fertilizer Amount (kg/ha)')
    plt.ylabel('Yield (kg/ha)')
    st.pyplot(fig)
    
    st.markdown("""
- **Interpretation** 📊
    - The scatterplot shows a negative linear relationship between rainfall (in mm) and yield (in kg/ha) 🌧️📉, as indicated by the downward-sloping red regression line.
    - Excessive rainfall might negatively impact yield 🌧️➡️❌, possibly due to over-saturation of soil or crop damage 🌱💧.
""")


    # Correlation Heatmap
    st.markdown("### 📊 Correlation Heatmap for Numeric Variables: 🔍")
    fig = plt.figure(figsize=(8, 6))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)
    
    st.markdown("""
- **Interpretation:**
  - 🌱 **Seed Variety** has the strongest positive correlation with yield (𝑟=0.68), indicating it is an important factor for yield prediction.  
  - 🚿 **Irrigation Schedule** (𝑟=0.56) and 🌾 **Fertilizer Amount** (𝑟=0.28) also positively influence yield.  
  - 🌧️ **Rainfall** has a weak negative correlation (𝑟=−0.25), confirming its potential adverse effects at high levels.  
  - ☀️ **Sunny Days** and 🌍 **Soil Quality** show negligible correlations (𝑟=0.10 and 𝑟=0.11).
""")


    # 3D Scatter Plot
    st.markdown("### 📊 3D Scatter Plot for Yield vs. Seed Variety & Irrigation Schedule: 🌱")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        data['Seed_Variety'], data['Irrigation_Schedule'], data['Yield_kg_per_hectare'],
        c=data['Yield_kg_per_hectare'], cmap='viridis', edgecolor='k', s=50
    )
    ax.set_xlabel("Seed Variety")
    ax.set_ylabel("Irrigation Schedule")
    ax.set_zlabel("Yield (kg/ha)")
    ax.set_title("3D Plot: Yield vs. Seed Variety & Irrigation")
    plt.colorbar(scatter, label='Yield (kg/ha)')
    st.pyplot(fig)

    st.markdown("""
- **Interpretation:**
  - 🌱 Yield increases as both seed variety improves (closer to 1.0) and irrigation frequency increases.
  - 🌾 Seed varieties closer to higher values (e.g., 0.8–1.0) combined with higher irrigation schedules (10–14) result in maximum yields.
  - 💧 Poor irrigation (e.g., <5) yields minimal output, regardless of seed variety.
""")
# Assuming 'data' is your dataset loaded in earlier steps

## 🛠️✨ Preprocessing: Cleaning and Preparing Data 🧹📊
# Displaying Outlier Treatment Visualization
    st.markdown("### 🧹 Outlier Treatment: Improving Model Accuracy by Handling Outliers")
    X, Y = data.drop(columns="Yield_kg_per_hectare"), data[["Yield_kg_per_hectare"]]
    plt.figure(figsize=(15, 4))
    sns.boxplot(data=data, palette="Set2", fliersize=5)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

# Function to replace outliers with mean
    def replace_outliers_with_mean(data):
     for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:  
            q1 = data[col].quantile(0.25)  
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1  

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Replace outliers with the mean of non-outlier values
            non_outlier_mean = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)][col].mean()

            data[col] = np.where((data[col] < lower_bound) | (data[col] > upper_bound), non_outlier_mean, data[col])
            return data

# Apply the outlier treatment
    data1 = replace_outliers_with_mean(X)

# Displaying boxplot after outlier treatment
    plt.figure(figsize=(15, 4))
    sns.boxplot(data=data1, palette="Set2", fliersize=5)
    plt.tight_layout()
    plt.show()

# Interpretation Section
    st.markdown("""
- **Interpretation**
1. **Soil_Quality**: Stable before, even less variable after replacing outliers with the mean.
2. **Seed_Variety**: No variation before or after.
3. **Fertilizer_Amount_kg_per_hectare**: High variability before, reduced after outlier treatment.
4. **Sunny_Days**: Consistent before, more consistent after outlier treatment.
5. **Rainfall_mm**: High variability before, reduced after outlier treatment.
6. **Irrigation_Schedule**: No variation before or after.
7. **Yield_kg_per_hectare**: High variability before, reduced after outlier treatment.
""")

# ⚖️📏 Standardizing: Scaling Features for Balanced Data 🌟
    st.markdown("### ⚖️📏 Standardizing: Scaling Features for Balanced Data 🌟")

    scaler = StandardScaler()
    X1 = pd.DataFrame(scaler.fit_transform(data1), columns=data1.columns)

# Show a sample of the scaled data
    st.write(X1.head())

# Interpretation of Scaling
    st.markdown("""
- **Interpretation**
- The data has been scaled using standardization 🔄 to ensure that no single feature dominates the model.
- Each feature is normalized to have a mean of 0 and a standard deviation of 1 📊, making the values comparable across different variables.
""")

# Hypothesis Testing Page
# Hypothesis Testing Page
elif page == "Hypothesis Testing":
    st.title("Hypothesis Testing: 🧠📊🔍")
    
    # Hypothesis Testing - Shapiro-Wilk Normality Test 🧪📊
    st.subheader("Shapiro-Wilk Normality Test 🧪📊")
    
    # Train-test split (assuming `data` is already loaded)
    X, Y = data.drop(columns="Yield_kg_per_hectare"), data[["Yield_kg_per_hectare"]]
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Display shapes of training and test data
    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Test data shape: {X_test.shape}")
    
    # Random Forest Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Model Prediction
    y_pred = rf.predict(X_test)
    
    # Flatten arrays
    y_test = y_test.values.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Perform Shapiro-Wilk test for normality of residuals
    from scipy.stats import shapiro
    stat, p_value = shapiro(residuals)
    st.write(f"Shapiro-Wilk Statistic: {stat}, P-value: {p_value}")
    
    # Interpretation of p-value
    if p_value < 0.05:
        st.write("The residuals are not normally distributed (reject H0).")
    else:
        st.write("The residuals are normally distributed (fail to reject H0).")
    
    # Plotting the Residuals
    st.subheader("Residuals Histogram")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title('Residuals Histogram')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Interpretation
    st.subheader("Interpretation")
    st.write("""
    - The Shapiro-Wilk test generates a test statistic close to 1 if the data are normally distributed.
    - Normally distributed residuals indicate that the model's errors are random and the model is unbiased.
    """)
    
    # Permutation Importance 📊🔍
    st.subheader("Permutation Importance 📊🔍")
    
    # Permutation Importance
    from sklearn.inspection import permutation_importance
    result = permutation_importance(rf, X, Y, n_repeats=10, random_state=42)
    
    # Plotting the Permutation Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(X.columns, result.importances_mean, color='skyblue')
    ax.set_xlabel('Permutation Importance')
    ax.set_title('Permutation Importance of Features')
    st.pyplot(fig)
    
    # Interpretation of Permutation Importance 📊🔍
    st.write("""
    **Interpretation of Permutation Importance:**
    
    Permutation Importance measures the decrease in model performance (e.g., R², accuracy) when the values of a particular feature are randomly shuffled, disrupting its relationship with the target variable. 
    
    - **🌾 Seed_Variety**: This feature has the highest importance, indicating it is the most critical predictor of yield. Changes to this feature have the greatest impact on model performance.
    - **💧 Irrigation_Schedule**: The second most important feature, suggesting that the irrigation plan significantly influences the target.
    - **☀️ Sunny_Days** and **🌱 Soil_Quality**: These have the least importance, implying they have a minimal effect on the model's predictions.
    """)
    
elif page == "Model Evaluation":
    st.title("**🚀 Model Comparison 📊: Accuracy Analysis ✅**")
    
    models = ['Linear Regression', 'Random Forest', 'XGBoost', 'SVM', 'KNN']
    accuracies = [0.9006, 0.9225, 0.9214, 0.8320, 0.872]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'orange', 'pink'])

    # Adding title and labels to the plot
    plt.title('Model Comparison: Accuracy', fontsize=16)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)

    # Set the y-axis limit
    plt.ylim([0.8, 1])

    # Display the plot in Streamlit
    st.pyplot(plt)

    st.write("""
             - **Interpretation**
             - **Random Forest** achieves the highest accuracy, followed by **XGBoost**, which shows a strong fit.
             - **Linear Regression** has moderate accuracy, performing better than **SVM** and **KNN** models.
             - **SVM** and **KNN** models have the lowest accuracy, suggesting they may require further tuning to improve performance.
    """)

    # Title for the Streamlit app
    st.title("**📊 Model Comparison: Cross-Validation Results & R² Scores for Regression Models 🔍**")

     
    # Define the models
    models = ['Linear Regression', 'Random Forest', 'XGBoost', 'SVM', 'KNN']
    Mean_accuracies = [ 0.8963, 0.9200, 0.9196, 0.8262,  0.8683]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'orange', 'pink'])

    # Adding title and labels to the plot
    plt.title('Model Comparison: Accuracy', fontsize=16)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel(' Mean Accuracy', fontsize=12)

    # Set the y-axis limit
    plt.ylim([0.8, 1])

    # Display the plot in Streamlit
    st.pyplot(plt)

   
    st.write("""
            - **Interpretation**
            - **Random Forest** shows the highest mean R² score (0.92 🎯), with a low standard deviation (0.0021 📊), indicating excellent consistency and strong model performance.
            - **XGBoost** has a similar mean R² score (0.92 🎯), with slightly higher variability (0.0027 📉), reflecting stable but slightly more fluctuating performance.
            - **Linear Regression** performs well (Mean R² = 0.90 🎯), with a small standard deviation (0.0045 📏), suggesting reliable predictions but less accuracy than the tree-based models.
            - **KNN** achieves a moderate mean R² (0.87 📈), with higher variability (0.0048 📊), indicating some instability but still reasonable performance.
            - **SVM** shows the lowest mean R² (0.83 ⚠️), with a standard deviation of 0.0039 📉, indicating relatively weak and inconsistent performance compared to the other models.

    """)
    
    st.title("### Hyperparameter Tuning Results 🎯🔧")
    st.write("""
            - **Fitting 5 folds for each of 324 candidates, totalling 1620 fits**
            - **Best Hyperparameters:**
            - `max_depth`: 30
            - `max_features`: 'sqrt'
            - `min_samples_leaf`: 1
            - `min_samples_split`: 2
            - `n_estimators`: 200
            - **Test R² (Best Model):** 0.92


         """)
    st.title(" 🎯 Random Forest Regressor: Model Evaluation After Hyperparameter Tuning 📊")
    st.write("""
            - **Interpretation**
            - **Accuracy: 91.97%** 🎯: The model predicts correctly 91.97% of the time.
            - **MSE: 3374.80** 📉: The average squared error between predicted and actual values.
            - **MAE: 46.11** 🔍: On average, the model's predictions are off by 46 units.
            - **RMSE: 58.09** 📊: The model's typical error is 58.09 units.


         """)
    
    # st.title("**🌲 Feature Importances in Random Forest Model 🔍: Visualizing Feature Impact**")
    # X, Y = data.drop(columns="Yield_kg_per_hectare"), data[["Yield_kg_per_hectare"]]
    # x_train, x_test , y_train ,y_test = train_test_split(X,Y,test_size=.20,random_state=42)
    # rfr = RandomForestRegressor(max_depth=30,max_features="sqrt",min_samples_leaf=1,min_samples_split=2,n_estimators=200)
    # rfr.fit(x_train,y_train)

    # pred = rfr.predict(x_test)
    # acc = r2_score(y_test,pred)
    # mse=mean_squared_error(y_test,pred)
    # mae = mean_absolute_error(y_test,pred) 
    # rmse = root_mean_squared_error(y_test,pred)
    # print(f"Accuracy : {acc}" ) 
    # print(f"MSE:{mse}")
    # print(f"MAE: {mae}")
    # print(f"RMSE:{rmse}")
     
    
    # feature_importances = rfr.feature_importances_
    # feature_names = x_train.columns
    # importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # importance_df = importance_df.sort_values(by='Importance', ascending=False)
    # plt.figure(figsize=(10,6))
    # plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    # plt.xlabel('Importance')
    # plt.ylabel('Features')
    # plt.title('Feature Importances in Random Forest Model')
    # plt.show()
    
    # st.write("""
    #         - **Interpretation**
    #         1. 🌱 **Seed_Variety** is the most important feature, having the highest impact on the model's predictions.
    #         2. 💧 **Irrigation_Schedule** holds significant influence on the model's output.
    #         3. 🌾 **Fertilizer_Amount_kg_per_hectare** contributes moderately to the model's predictions.
    #         4. 🌧️ **Rainfall_mm** plays a smaller, but still notable role in the model.
    #         5. 🦠 **Soil_Quality** has minimal importance in the model.
    #         6. ☀️ **Sunny_Days** is the least important feature in the Random Forest Model.


    #      """)
    
    st.title("**🌲 Feature Importances in Random Forest Model 🔍: Visualizing Feature Impact**")

# Assume 'data' is a DataFrame already loaded
# Example: data = pd.read_csv("your_data.csv")
# Replace this with your actual data loading code
# data = ...

# Split data into features and target
    X, Y = data.drop(columns="Yield_kg_per_hectare"), data["Yield_kg_per_hectare"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Train the Random Forest Regressor
    rfr = RandomForestRegressor(
        max_depth=30,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
)
    rfr.fit(x_train, y_train)

# Predictions and evaluation metrics
    pred = rfr.predict(x_test)
    acc = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)

# Display metrics in Streamlit
    st.write(f"**Accuracy (R2 Score):** {acc:.4f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")

# Feature importance
    feature_importances = rfr.feature_importances_
    feature_names = x_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importances in Random Forest Model')

# Display the plot in Streamlit
    st.pyplot(fig)
 
    st.write("""
            - **Interpretation**
             1. 🌱 **Seed_Variety** is the most important feature, having the highest impact on the model's predictions.
             2. 💧 **Irrigation_Schedule** holds significant influence on the model's output.
             3. 🌾 **Fertilizer_Amount_kg_per_hectare** contributes moderately to the model's predictions.
             4. 🌧️ **Rainfall_mm** plays a smaller, but still notable role in the model.
             5. 🦠 **Soil_Quality** has minimal importance in the model.
             6. ☀️ **Sunny_Days** is the least important feature in the Random Forest Model.


          """)

    st.title("**🔍SHAP Feature Importance & Insights 📊**")


# Assume x_train, y_train, x_test, y_test are defined
# Fit the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)

# Make predictions on the test set
    rf_pred = rf_model.predict(x_test)

# Evaluate model performance
    rf_acc = r2_score(y_test, rf_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(rf_mse)

# Display model evaluation metrics in Streamlit
    st.write(f"Random Forest Regression Accuracy (R2): {rf_acc}")
    st.write(f"Random Forest MSE: {rf_mse}")
    st.write(f"Random Forest MAE: {rf_mae}")
    st.write(f"Random Forest RMSE: {rf_rmse}")

# Reduce the number of background samples for SHAP
    background_data = shap.kmeans(x_train, 10)

# Create a SHAP explainer
    explainer = shap.KernelExplainer(rf_model.predict, background_data)

# Get SHAP values for the test set
    shap_values = explainer.shap_values(x_test)

# Plot and render SHAP summary plot in Streamlit
    st.subheader("SHAP Summary Plot")
    import matplotlib.pyplot as plt  # Ensure matplotlib is imported

# Render the SHAP summary plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, x_test, show=False)  # Avoid direct display
    st.pyplot(fig)

# Plot feature importance using SHAP (bar plot)
    st.subheader("SHAP Feature Importance (Bar Plot)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    st.pyplot(fig)
 
    st.write("""
           - **Interpretation**
- 🌊 Irrigation_Schedule: High values have a significant positive impact on the model output.
- 🌱 Seed_Variety: Different varieties have varying impacts, with some having a strong positive effect.
- 💧 Fertilizer_Amount_kg_per_hectare: Higher amounts generally increase the model output.
- 🌧️ Rainfall_mm: More rainfall tends to have a positive impact, but the effect varies.
- 🌍 Soil_Quality: Better soil quality has a moderate positive impact.
- ☀️ Sunny_Days: More sunny days have a smaller positive impact.

          """)


    st.title("### 📊 PDP (Partial Dependence Plot) for Feature Importance 🚀")
    from sklearn.inspection import PartialDependenceDisplay 
    X, Y = data.drop(columns="Yield_kg_per_hectare"), data["Yield_kg_per_hectare"]

# Split data
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

# Choose the features for which you want to plot the PDP
    features = [0, 1, 2, 3, 4, 5]  # Indices of features to plot

# Plot Partial Dependence
    fig, ax = plt.subplots(figsize=(10, 8))
    PartialDependenceDisplay.from_estimator(model, X_train, features=features, grid_resolution=50, ax=ax)

# Display the plot in Streamlit
    st.pyplot(fig)
    st.write("""
         - **Interpretation**
1. 🌱 **Soil Quality**: Better soil quality leads to a slight increase in crop yield.
2. 🌾 **Seed Variety**: Different seed varieties have a significant impact on crop yield.
3. 🌿 **Fertilizer Amount**: More fertilizer gradually increases crop yield.
4. ☀️ **Sunny Days**: More sunny days slightly improve crop yield.
5. 🌧️ **Rainfall**: Higher rainfall slightly decreases crop yield.
6. 🚿 **Irrigation Schedule**: Optimal irrigation schedules significantly boost crop yield.""")

    st.title("### 📉 Learning Curve: Model Performance Analysis After Hyperparameter Tuning 🔍")
    from sklearn.model_selection import learning_curve
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

# Assuming rfr, x_train, and y_train are already defined
    train_sizes, train_scores, test_scores = learning_curve(
    rfr, x_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Calculate the mean and standard deviation of the training and test scores
    train_mean = np.mean(-train_scores, axis=1)
    test_mean = np.mean(-test_scores, axis=1)
    train_std = np.std(-train_scores, axis=1)
    test_std = np.std(-test_scores, axis=1)

# Create the plot
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, label='Training Error', color='blue')
    ax.plot(train_sizes, test_mean, label='Test Error', color='red')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='red', alpha=0.1)
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Learning Curve')
    ax.legend()

# Display the plot in Streamlit
    st.pyplot(fig)


    st.write("""
         - **Interpretation**
- 📉 The training error remains consistently low, indicating the model fits the training data well.
- 📈 The test error decreases as the training size increases, suggesting the model generalizes better with more data.
- 🔍 The gap between training and test error narrows with larger training sizes, indicating reduced overfitting.""")

    st.title("### 🔍 Residuals vs Predicted Values 📉")
# Make predictions on the validation set
    import matplotlib.pyplot as plt
    import streamlit as st

# Assuming y_pred and y_val are already defined
    y_pred = model.predict(X_val)

# Ensure y_val is a 1D array, if it's a DataFrame, select the column
    y_val = y_val.squeeze()  # This makes sure y_val is 1D if it's a DataFrame

# Calculate residuals (difference between actual and predicted values)
    residuals = y_val - y_pred

# Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')  # Zero error line
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')

# Display the plot in Streamlit
    st.pyplot(plt)


    st.write("""
         - **Interpretation**
- 📉 Random Scatter: Residuals are randomly scattered around zero, indicating a good model fit.
- ✅ No Pattern: No clear pattern in residuals, suggesting the model's assumptions are likely met.
- 🔍 Consistent Spread: Residuals spread consistently across predicted values, implying homoscedasticity.""")

#     st.title(" #####📈 Model Evaluation: MSE & R² Analysis 🔍")
#     X, Y = data.drop(columns="Yield_kg_per_hectare"), data["Yield_kg_per_hectare"]
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Initialize and train the Random Forest model
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

# # Predictions
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)

# # Evaluate performance
#     train_mse = mean_squared_error(y_train, y_train_pred)
#     test_mse = mean_squared_error(y_test, y_test_pred)

#     train_r2 = r2_score(y_train, y_train_pred)
#     test_r2 = r2_score(y_test, y_test_pred)

# # Print results
#     print(f"Training MSE: {train_mse:.2f}")
#     print(f"Validation/Test MSE: {test_mse:.2f}")
#     print(f"Training R²: {train_r2:.2f}")
#     print(f"Validation/Test R²: {test_r2:.2f}")

# # Check for overfitting or underfitting
#     if train_r2 > 0.9 and (train_r2 - test_r2) > 0.1:
#         print("The model is overfitting: It performs well on training data but poorly on test data.")
#     elif train_r2 < 0.7 and test_r2 < 0.7:
#         print("The model is underfitting: It performs poorly on both training and test data.")
#     else:
#         print("The model has a good fit: It generalizes well to unseen data.")

#     st.write("""- The model shows excellent training performance with a high R² value 📈💯 """)
elif page == "Model":
    

    
    import streamlit as st
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

# Sample data (replace with your actual dataset)
# Generate a sample dataset for demonstration

# Convert data into DataFrame
    df = pd.DataFrame(data)

# Define the Random Forest model training function
    def train_model():
        X = df.drop(columns='Yield_kg_per_hectare')
        Y = df['Yield_kg_per_hectare']

    # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model

# Load the trained model
    model = train_model()

# Streamlit application
    def app():
        st.title('Crop Yield Prediction')

    # User inputs
        soil_quality = st.number_input('Soil Quality', min_value=0.0, max_value=100.0, value=80.0)
        seed_variety = st.selectbox('Seed Variety', options=[0, 1], index=1)
        fertilizer_amount = st.number_input('Fertilizer Amount (kg per hectare)', min_value=0.0, value=100.0)
        sunny_days = st.number_input('Sunny Days', min_value=0, value=100)
        rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=500.0)
        irrigation_schedule = st.selectbox('Irrigation Schedule', options=[1, 2, 3, 4, 5, 6, 7, 8], index=3)

    # Button for prediction
        if st.button('Predict'):
        # Prepare the input data for prediction
            input_data = np.array([[soil_quality, seed_variety, fertilizer_amount, sunny_days, rainfall, irrigation_schedule]])

        # Prediction
            prediction = model.predict(input_data)

        # Display the result
            st.write(f'Predicted Yield (kg per hectare): {prediction[0]:.2f}')

    if __name__ == "__main__":
        app()
