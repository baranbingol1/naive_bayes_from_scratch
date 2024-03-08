from naive_bayes import predict, train, eval
import pandas as pd

data = pd.read_csv("diabetes.csv")

# veriyi bölelim
train_data = data.sample(frac=0.8, random_state=42)
val_data = data.drop(train_data.index)

class_stats = train(train_data)

acc = eval(class_stats, val_data)
print(f"Accuracy : {acc}") # Accuracy : 0.7662337662337663