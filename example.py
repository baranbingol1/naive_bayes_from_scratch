from naive_bayes import train, eval
import pandas as pd

# örnek veri
data = pd.read_csv("diabetes.csv")

# veriyi bölelim
train_data = data.sample(frac=0.8, random_state=42)
val_data = data.drop(train_data.index)

class_stats = train(train_data)

acc = eval(class_stats, val_data, 1e-10)
print(f"Accuracy : {acc}") # Accuracy : 0.7662337662337663
