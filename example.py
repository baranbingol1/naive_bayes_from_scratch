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

acc = eval(class_stats, val_data, var_smoothing=1e-5)
print(f"Accuracy : {acc}") # Accuracy : 0.7662337662337663

acc = eval(class_stats, val_data, var_smoothing=1)
print(f"Accuracy : {acc}") # Accuracy : 0.7532467532467533

acc = eval(class_stats, val_data, var_smoothing=10)
print(f"Accuracy : {acc}") # Accuracy : 0.7337662337662337
