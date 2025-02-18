import pandas as pd
import kagglehub
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("C://tested.csv")  #Путь к исходному файлу
print(df.head(50))
df["Age"] = df["Age"].fillna(df["Age"].median()) #Заполняем пустые значения медианой
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0]) #Модой

scaler = MinMaxScaler()
df["Age"] = scaler.fit_transform(df[["Age"]])#Нормализация
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True) #кодирование столбца Embarked

print(df.head(50))
df.to_csv("processed_titanic.csv", index=False)