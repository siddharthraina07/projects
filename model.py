import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the csv file
df = pd.read_csv("Downloads/deployment/iris.csv")


print(df.head())

# Select independent and dependent variable
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling step
    ('rf', RandomForestClassifier())  # Random Forest model step
])

# Fit the model
pipeline.fit(X_train, y_train)

example_set=[[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2]]

print(' example prediction ')
print(pipeline.predict(example_set))
    
# Make pickle file of our model

joblib.dump(pipeline, "model1.pkl")



# Feature scaling
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test= sc.transform(X_test)


# Instantiate the model
#classifier = RandomForestClassifier()
