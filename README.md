Link to code: Link to code: https://github.com/MelodySiska/Berkeley-Machine-Learning-and-Artificial-Intelligence/blob/main/SiskaUsedCarPracticalApplication2.ipynb  

# Used Car Prices Practical Application 2
 Analysis to understand which factors impact used car prices

 ## Overview:
In this project, I delved into a dataset sourced from Kaggle. Originally, the dataset comprised data on 3 million used cars. For the purposes of this analysis and to enhance processing  speed, I  worked with a subset containing information on 426,000 cars. The objective is to identify the factors that influence a car's price. Based on my analysis, I was asked tol provide the client, a used car dealership, with clear recommendations on what features consumers prioritize when purchasing a used car.


## Business Understanding: 
This data analysis is looking at a large dataset to understand and advise my customer, a used car dealer, on what sorts of features impact used car prices. From a data perspective, the task is to develop predictive models to identify and quantify the key factors that influence the prices of used cars. This involves preprocessing and cleaning the data, potentially engineering relevant features, and applying machine learning algorithms to create models that can accurately predict used car prices based on various attributes such as age, mileage, condition, and manufacturer. The goal is to optimize the model's performance through cross-validation and hyperparameter tuning to ensure it generalizes well to unseen data.

## Data Understanding: 
### Insights from initial data summaries:
Based on the Non-null Count, some columns have many missing values, especially condition, cylinders, drive, size, and VIN. There are a mix of numerical and categorical data types.

### From the statistical summary:
**id**:
- Seems to be a unique identifier for each entry and not relevant for analysis. Plan to drop.
**price**:
- Range: 0 to 3,736,929,000 (extremely high maximum value and low value of 0, strongly suggesting outliers or more than likely errors)
- Mean: 75,199.03 (probably so high because of the super high data points)
- Median (50th percentile): 13,950 (indicates a right-skewed distribution)
**year**:
- Range: 1900 to 2022 (likely includes vintage cars or could be errors)
- Mean: 2011.24
- Median: 2013
**odometer**:
- Range: 0 to 10,000,000 (another potential for outliers or error - super high and used cars with 0 mileage suspect)
- Mean: 98,043.33 (Again potentially skewed by the large numbers in range)
- Median: 85,548

### Looking at columns missing values and unique values and cardinality
- The 'id' column is a unique identifier and not relevant for analysis. It should be dropped.
- Likewise the 'VIN' column is another identifer and not relevant for analysis and many missing values. It should be dropped.
- Columns like 'condition', 'cylinders', 'fuel', 'drive', 'size', 'type', and 'paint_color' have manageable levels of cardinality and missing values and could be important for analysis.
- 'year', 'manufacturer', 'model', 'odometer', 'title_status', 'transmission', and 'state' appear to have significant potential for analysis, though some have missing values that need handling.

### Insights from the Distribution of Numerical Features
![](images/DistributionOfNumericalFeatures.png)   
- **id**:
  - The `id` feature is a unique identifier for each entry.
  - Not relevant for analysis and can be dropped.

- **price**:
  - The `price` feature exhibits a highly right-skewed distribution.
  - There are many low-priced entries, but a few extremely high values indicate potential outliers or errors.
  - Action: Cap outliers for better analysis

- **year**:
  - The `year` feature shows most entries are from the late 1990s to 2022.
  - There are some entries with very old years, potentially indicating vintage cars or data entry errors.
  - Action: Filter out extremely old years
  
- **odometer**:
  - The `odometer` feature also shows a highly right-skewed distribution.
  - Many entries have low odometer readings, but there are some extremely high values, which could be errors.
  - Action: Similar to `price`, cap outliers for better analysis
  

### Next Steps:
1. **Drop the `id` and `vin` columns** since they do not add value to the analysis.
2. **Handle outliers** in the `price` and `odometer` columns to improve the model's performance.
3. **Filter the `year` column** to remove unlikely or irrelevant years (e.g., extremely old years).
4. Continue with further exploratory data analysis and data preparation steps.

### Insights from the Correlation Matrix:
![](images/CorrelationMatix.png)   
- **Price**:
  - Has a very weak negative correlation with `year` (-0.0049), indicating that newer cars do not significantly influence the price in our dataset.
  - Has a very weak positive correlation with `odometer` (0.0100), suggesting that the odometer reading (mileage) does not strongly influence the price.
  
- **Year**:
  - Shows a weak negative correlation with `odometer` (-0.1572), indicating that newer cars tend to have lower mileage, which is intuitive.
  
- **Odometer**:
  - Exhibits a very weak positive correlation with `price` (0.0100), reinforcing the observation that mileage does not significantly impact the price in our dataset.

### Overall Observations:
- The `id` column, being a unique identifier, does not have any meaningful correlation with other variables and should be excluded from the analysis.
- The correlations between `price`, `year`, and `odometer` are very weak, suggesting that other features (likely categorical) may play a more substantial role in determining the price but by eliminating the rows with the hughe outliers and less than $1000 I anticipate this may increase.
- This weak correlation indicates a need for further exploration and inclusion of other variables to build a more predictive model for used car prices.

## Insights from Categorical Features Analysis:

### Region
![](images/DistributionOfRegion.png) 
- There are 404 unique regions.
- The distribution is quite broad, with some regions having significantly more entries than others.
- Some regions are represented more frequently, potentially indicating larger markets or more active listings in those areas.

### Manufacturer
![](images/DistributionOfManufacturer.png) 
- Ford, Chevrolet, and Toyota are the top three most common manufacturers.
- There's a long tail of less common manufacturers, some. like Morgan and Alfa-Romeo, having very few entries.
- The majority of the cars in the dataset are from popular and commonly known brands that I have actually heard of :-)

### Condition
![](images/DistributionOfCondition.png) 
- Most cars are in 'good' and 'excellent' condition.
- 'New' cars are very few, likely because the dataset focuses on used cars. This should probably be excluded from the dataset.
- The 'salvage' condition is the least common, which makes sense as these cars are typically less desirable.

### Cylinders
![](images/DistributionOfCylinders) 
- Most cars have 4, 6, or 8 cylinders.
- There are very few cars with 3, 5, 10, or 12 cylinders, indicating these configurations are less common at least I have never heard of them.

### Fuel
![](images/DistributionOfFuel.png) 
- The majority of cars run on gas.
- Diesel and 'other' fuels are less common.
- Hybrid and electric cars are rare in this dataset, which may reflect the market trends at the time the data was collected.

### Title Status
![](images/DistributionOfTitleStatus.png) 
- Most cars have a 'clean' title status.
- Rebuilt and salvage titles are less common.
- Other title statuses like 'lien', 'missing', and 'parts only' are very rare.

### Transmission
![](images/DistributionOfTransmission.png) 
- Automatic transmission is the most common type.
- Manual transmissions are less common but still significant.
- 'Other' transmission types are relatively few, I am not sure what this could be if it isnt automatic or manual but that could be my lack of car knowledge. 

### Next Steps
- Drop the some of these columns based on relevance or similar columns like region and state 
- Drop new cars from the dataset

### Insights from Initial Data Quality Checks

#### Number of Duplicate Rows:
- **Duplicates**: 
  - There are no duplicate rows in the dataset. 

#### Consistency Checks:
- **Negative Odometer Values**: 
  - There are no entries with negative odometer values. This ensures that all odometer readings are logical and represent real-world scenarios although the 0s and low readings for used cars are a bit suspicous.

- **Unreasonable Year Values**: 
  - There are no entries with unreasonable year values (outside the range of 1900 to the current year). This confirms that all entries have realistic manufacturing years, adding credibility to the dataset but for our purposes I am planning to remove vintage cars and new cars from the dataset.

### Next Steps
1. **Remove columns and rows that do not make sense based on previous statements**: 
2. **Handle Blank Values**: 
3. **Consider Feature Engineering**: Create new features or transform existing ones.
4. **Encoding Categorical Variables**: Convert categorical variables into numerical formats suitable for modeling.
5. **Scaling/Normalization**: Normalize or scale the numeric features to ensure they're on a similar scale.

## Data Preparation
After the initial exploration and fine tuning of the business understanding, it is time to construct the final dataset prior to modeling.  I chose to drop columns with more than 50% missing data, and dropped columns 'VIN', 'region', 'model', 'id' which didnt add value to the analysis. I dropped rows that were missing the year, had prices that are $1000 or less, without a listed manufacturer, odometer readings less 1,000 or more than 250,000.  I alos removed  high price outliers above the 99th percentile. As a result, there were 346,125 vehichles with 13 features in the dataset.

### Insights from Price Distribution After Removing High Outliers and other cleaning
![](images/DistributionOfPriceAfterRemovingHighOutliers.png) 

- **Histogram Analysis**:
  - The updated histogram shows a more normalized distribution of car prices compared to the previous histogram.
  - Most prices now fall within the range of $0 to $60,000, with a noticeable peak around $10,000.
  - The skewness present in the earlier histogram, due to extremely high outliers, is significantly reduced.
  - The distribution now has a more gradual decline after the peak.

- **Comparison to Initial Histogram**:
  - **Initial Histogram**:
    - Had extreme values reaching up to $3.7 billion, which heavily skewed the distribution.
    - Most of the data was compressed into the lower end of the price range, making it difficult to observe the true distribution of prices.
    - The high outliers overshadowed the majority of data points.
  - **Updated Histogram**:
    - By capping the prices, the extreme outliers are removed, resulting in a clearer and more representative histogram.
    - The data is now more evenly spread across the range, allowing for better insights.

#### Column Names:
- The dataset has 13 columns after removing irrelevant columns and columns with excessive missing values.

#### Missing Values:
- Significant missing values exist in several columns, notably `condition`, `cylinders`, `drive`, `type`, and `paint_color`.
- I will handle missing values through imputation or stating unknown.

#### Unique Values:
- The `price`, `odometer`, and `year` columns have high unique value counts, which makes sense.
- The columns `manufacturer` and `state` have a a large number of unique values, Iwill group them into areas for state and highend and other for manufacturer.
- The remaining Categorical columns `condition`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `type`, and `paint_color` seem to have a manageable number of unique values, suitable for encoding and further analysis.

### Data Preparation: Grouping States and Manufacturers

In order to simplify the analysis and modeling process, I grouped the states into broader regions and classified manufacturers into high-end and other categories.

#### State Groupings

I grouped the states into the following regions:

- **East Coast**: ['CT', 'DE', 'FL', 'GA', 'MA', 'MD', 'ME', 'NC', 'NH', 'NJ', 'NY', 'PA', 'RI', 'SC', 'VA', 'VT']
- **West Coast**: ['CA', 'OR', 'WA']
- **Midwest**: ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI']
- **South**: ['AL', 'AR', 'KY', 'LA', 'MS', 'OK', 'TN', 'TX', 'WV']
- **Other**: ['AK', 'AZ', 'CO', 'DC', 'HI', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY']

Any state not in these groups was assigned to the 'Unknown' category.

#### Manufacturer Groupings

Manufacturers were classified into high-end and other categories based on my (common?) perceptions of luxury and market positioning. The high-end manufacturers include:

- **High-End**: ['audi', 'bmw', 'cadillac', 'jaguar', 'land rover', 'lexus', 'mercedes-benz', 'porsche', 'rover', 'tesla', 'aston-martin', 'volvo', 'infiniti', 'lincoln', 'ferrari']

All other manufacturers were grouped into the 'Other' category.

## Insights from Price Distribution Analysis

### Price Distribution by Area of Country

![](images/BoxPlotPriceDistributionByAreaOfCountry.png) 

- **South**: The median price of cars in the South is around $20,000. The interquartile range (IQR) is relatively wide, indicating a diverse range of car prices. There are a few outliers above $60,000.
- **Other**: The price distribution in the Other category is similar to the South, with a median around $20,000 and a wide IQR. The presence of outliers above $60,000 is also noticeable.
- **West Coast**: The median price on the West Coast is slightly lower, around $18,000. The IQR is also wide, with outliers similar to other regions.
- **East Coast**: The median price on the East Coast is around $19,000. The IQR is wide with a few outliers above $60,000.
- **Midwest**: The Midwest has the lowest median price among the regions, around $17,000. The IQR is wide, with outliers present.

### Price Distribution by Manufacturer Category

![](images/BoxPlotPriceDistributionByManufacturerCategory.png)  

- **Other**: The median price for cars in the Other category is around $15,000. The IQR is wide, showing a diverse range of car prices. There are a few outliers above $60,000.
- **High-End**: The median price for High-End cars is significantly higher, around $30,000. The IQR is also wide, indicating a broad range of prices. There are several outliers above $60,000.

### Key Takeaways

- The price distributions across different regions do not show significant variations in the median values. However, there are outliers present in all regions.
- High-End manufacturers generally have a higher median price compared to other manufacturers, which aligns with expectations given the premium nature of these brands.
- The wide IQR in all categories indicates a significant variation in car prices within each group, suggesting a diverse dataset.

## Odometer Imputation Strategy
![](images/PlotPriceDistributionByOdometer.png)  
Given the skewed distribution, I proceeded with the median for imputing missing odometer values.


### Model Performance Summary
I Split the dataset into training and testing sets and ran multiple models to learn the best way to analyze impacts of the features on pricing.
**Linear Regression**:
- RMSE: 7592.0216
- R2: 0.6566

**Ridge Regression**:
- RMSE: 7592.0184
- R2: 0.6566

**Lasso Regression**:
- RMSE: 7592.5990
- R2: 0.6565

**Elastic Net Regression**:
- RMSE: 9030.8559
- R2: 0.5141

**Random Forest**:
- RMSE: 3973.0588
- R2: 0.9060

###Insights

**Linear Models**:
- Linear, Ridge, and Lasso regression models all perform similarly with RMSE around 7592 and R2 around 0.6565-0.6566. This suggests that adding regularization (Lasso and Ridge) does not significantly impact performance for this dataset.
- The performance indicates that these models are able to explain approximately 65.66% of the variance in the price, but their prediction errors are relatively high.

**Elastic Net Regression**:
- This model performs worse than the other linear models, with a higher RMSE and lower R2. This suggests that the combination of L1 and L2 regularization in Elastic Net does not suit this dataset as well as the other linear approaches.

**Random Forest**:
- This model significantly outperforms all the linear models with a much lower RMSE of 3973.0588 and a higher R2 of 0.9060.
- The Random Forest model explains approximately 90.60% of the variance in the price and has a much lower prediction error, indicating it is better at capturing the underlying patterns in the data.

###Issues Processing due to onehot encoding etc increasing the nukber of columns
I ran intoissues trying to run the Grid Search cross-validation due to the number of features and columns. I let it run overnight and still was unable to complete the analysis in this manner despite decreasing the number of folds. As a result, I decided to work on feature reduction. I wqs able to reduce from Original number of features of 73  to a Reduced number of features of 8 ['year' 'odometer' 'cylinders_4 cylinders' 'cylinders_8 cylinders' 'fuel_diesel' 'drive_4wd' 'drive_fwd' 'manufacturer_category_Other']. 

## Insights from Feature Importances - Pareto Chart

The Pareto chart below highlights the importance of various features in predicting the price of used cars. The key insights are as follows:

1. **Year**:
   - The year of the car is the most significant predictor of its price, contributing approximately 45% to the model's decision-making process.
   - Newer cars tend to have higher prices, which aligns with common expectations.

2. **Odometer**:
   - The mileage (odometer reading) is the second most important feature, accounting for 15% of the importance.
   - Cars with lower mileage generally have higher prices, reflecting their better condition and less wear and tear.

3. **Drive Type (FWD and 4WD)**:
   - The drive type (front-wheel drive and four-wheel drive) together contribute around 10% to the feature importance.
   - This indicates that the drive type has a noticeable impact on the car's price, with certain types being more desirable based on the region and usage.

4. **Fuel Type (Diesel)**:
   - Diesel fuel type contributes approximately 8% to the feature importance.
   - Diesel cars may have different pricing dynamics due to their fuel efficiency and performance characteristics.

5. **Cylinders**:
   - The number of cylinders (6 cylinders and 4 cylinders) together account for about 5% of the importance.
   - This suggests that engine size and configuration are relevant factors in determining car prices.

6. **Manufacturer Category (Other)**:
   - The manufacturer category (high-end vs. other) has a small but notable influence, contributing around 1% to the importance.
   - High-end manufacturers generally command higher prices, reflecting the premium nature of these brands.

### Key Takeaways

- **Significant Factors**: The year and odometer reading are the most influential factors in determining used car prices. Ensuring accurate and updated information on these attributes is crucial for pricing strategies.
- **Drive and Fuel Type**: The drive and fuel type also play a significant role, indicating that cars with specific configurations may be more valuable in certain markets.
- **Engine Size and Manufacturer**: While less influential than the top factors, engine size and manufacturer still contribute to the overall pricing model, highlighting the importance of these features in specific contexts.

These insights can help used car dealers better understand the key drivers of car prices and make informed decisions about their inventory and pricing strategies.

![Feature Importances - Pareto Chart](images/ParetoChartOfKeyReducedFeatures.png) 

### Model Evaluation Summary

**R-squared Value**:
The R-squared value of our final Random Forest model, which uses the selected features, is **0.8522**.

**Percentage of Variation Explained**:
The model explains **85.22%** of the variation in used car prices based on the features selected.

### Explanation

The R-squared value (coefficient of determination) indicates the proportion of the variance in the dependent variable (used car prices) that is predictable from the independent variables (selected features such as year, odometer, etc.). An R-squared value of 0.8522 means that 85.22% of the variability in used car prices can be explained by the features included in our model. This is a strong indication that the model is effective at capturing the relationships between the predictors and the target variable.

### Insights

- **Year** and **odometer** are the most influential features in determining the price of used cars, with significant importance scores in our model.
- Other features, like **drive type (fwd)**, **fuel type (diesel)**, and the number of **cylinders**, also contribute to the model but to a lesser extent.
- The high R-squared value suggests that the selected features are highly relevant and sufficient for building an accurate predictive model for used car prices.

### Conclusion

The Random Forest model, enhanced with feature selection, provides a robust and reliable prediction of used car prices, explaining over 85% of the variance in the data. This high level of explained variation indicates that the model effectively captures the key drivers of used car prices, making it a valuable tool for used car dealers to fine-tune their inventory and pricing strategies.

### Evaluation and Recommendations

**1. Review of Business Objective**
Our primary goal was to identify the key factors that influence used car prices. By understanding these factors, we aim to help you price your inventory competitively and make informed decisions.

**2. Model Performance**
We tested several approaches to predict car prices, including basic models and more advanced techniques. Here are the results:

- **Basic Models**: These models explained about 65% of the variation in car prices. However, they had higher prediction errors.
- **Advanced Model (Random Forest)**: This model performed the best, explaining about 91% of the variation in car prices and providing much more accurate predictions.

**3. Key Drivers of Used Car Prices**
From our best-performing model, we identified several important factors that influence used car prices:
- **Year**: Newer cars tend to be priced higher.   
- **Mileage**: Cars with lower mileage are generally more valuable.
- **Condition**: Cars in 'excellent' or 'like new' condition fetch higher prices.
- **Brand**: High-end brands like BMW, Mercedes-Benz, and Tesla command premium prices.
- **Type**: SUVs and trucks tend to be priced higher compared to other types like sedans and hatchbacks.

**4. Revisiting Earlier Phases**
Given our findings, we recommend:
- **Feature Engineering**: Explore additional features or interactions between existing features to further improve model performance.
- **Hyperparameter Tuning**: Perform more extensive hyperparameter tuning with a smaller, more manageable parameter grid to optimize the Random Forest model.
- **Model Evaluation**: Cross-validate with additional metrics such as Mean Absolute Error (MAE) or Mean Absolute Percentage Error (MAPE) to provide a more comprehensive evaluation.

**5. Conclusion and Recommendations**
The Random Forest model provided meaningful insights into the drivers of used car prices, aligning well with our business objectives. We can confidently bring back the following information to our client:
- Key features impacting prices, allowing the dealership to make informed pricing decisions.
- High-performing model that can be used for future price predictions.
- Recommendations for further improvements through additional feature engineering and tuning.

### Deployment

**Introduction:**
The goal is to provide used car dealers with actionable insights and recommendations that will help them optimize their inventory and pricing strategies.

**1. Summary of Findings:**
Through the detailed analysis, I have identified the key factors that influence used car prices. The best-performing model, the Random Forest, explains 85.22% of the variation in car prices, indicating a strong predictive capability. Key features impacting used car prices include:

- **Year:** Newer cars tend to have higher prices.
- **Odometer (Mileage):** Cars with lower mileage are generally more valuable.
- **Condition:** Cars in 'excellent' or 'like new' condition fetch higher prices.
- **Drive Type:** Certain drive types like 'fwd' (front-wheel drive) influence prices significantly.
- **Fuel Type:** Diesel cars show a distinct pricing pattern.
- **Cylinders:** The number of cylinders in a car’s engine also impacts the price.
- **Manufacturer Category:** High-end brands such as BMW, Mercedes-Benz, and Tesla command premium prices.

**2. Detailed Report:**
Here is a detailed account of the methodology and findings:

- **Business Objective:** To identify the key drivers of used car prices to help dealers optimize their inventory pricing strategies.
- **Data Preprocessing:** I cleaned and preprocessed the data, handling missing values and encoding categorical variables.
- **Modeling:** I tested several models, including Linear Regression, Ridge Regression, Lasso Regression, Elastic Net Regression, and Random Forest. The Random Forest model outperformed the others.
- **Model Performance:** The Random Forest model achieved an RMSE of 3973.06 and an R-squared value of 0.9060 before feature selection, indicating it explains 90.60% of the variation in car prices. After feature selection, the model still performed well with an R-squared value of 0.8522, explaining 85.22% of the variation in car prices.

**3. Key Drivers of Used Car Prices:**
From the best-performing model, I identified several important factors that influence used car prices:

- **Year (0.45):** This is the most significant factor. Newer cars tend to have higher prices.
- **Odometer (0.15):** Lower mileage is associated with higher prices.
- **Drive Type (fwd, 0.08):** Certain drive types influence car prices.
- **Fuel Type (diesel, 0.07):** Diesel cars show a distinct pricing pattern.
- **Cylinders (8 cylinders, 0.03):** The engine configuration impacts the price.
- **Manufacturer Category (High-End, 0.01):** High-end brands command premium prices.

**4. Recommendations:**
Based on the findings, I recommend:

- **Focus on Key Features:** Pay close attention to the car's year, mileage, and condition when pricing inventory.
- **Leverage the Model:** Use the Random Forest model for future price predictions to maintain competitive pricing.
- **Feature Engineering:** Explore additional features or interactions between existing features to further improve model performance.
- **Hyperparameter Tuning:** Perform more extensive hyperparameter tuning with a smaller, more manageable parameter grid to optimize the Random Forest model.

**5. Model Deployment:**
To implement and utilize our model, follow these steps:

- **Implementation:** Use the provided code snippets to preprocess new data and make price predictions using the trained Random Forest model.
- **Integration:** Integrate the model into your existing systems via a user-friendly interface or an API.
- **Maintenance:** Regularly update the model with new data to ensure its predictions remain accurate.

**Training and Support:**
Offer training sessions and documentation to help your team understand and utilize the model effectively:

- **Workshops:** Conduct workshops or webinars to explain how the model works and how to interpret its predictions.
- **User Manual:** Provide a user manual detailing the steps for preprocessing data, running the model, and interpreting the results.

**Conclusion:**
The deployment phase ensures that the insights and models developed are effectively communicated and utilized by the car dealships. By providing a detailed report, actionable recommendations, and practical deployment steps, this can empower used car dealers to make data-driven decisions and optimize their inventory management.

