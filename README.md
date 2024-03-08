# naive_bayes_from_scratch

Naive Bayes algoritmasının sıfırdan minimal inşa edilimidir. NumPy, Pandas ve SciPy gibi standart kütüphanelerin bazı fonksiyonları veri yükleme, olasılık hesaplama vs. gibi temel işlemler için kullanılmıştır.

Kullanım :

```python 
from naive_bayes import train, eval

# train_data ve val_data bir pandas dataframe'i olmalıdır ve son sütunlarında label sütunu bulunmalıdır. Labellar 0 ve 1 gibi nümerik değerler veya string olabilir. Algoritma binary veya multiclass prediction yapabilir.
train_data = pd.read_csv("train.csv")
val_data = pd.read_csv("val.csv")

class_stats = train(train_data)
acc = eval(class_stats, val_data, var_smoothing=1e-10)
print(f"Doğruluk oranı: {acc}")
```

Modelin tek hiperparametresi temelde sıfıra bölüm hatasını engellemek için kullanılan var_smoothing değeridir. Default değer genel olarak iyi çalışmaktadır ve 1e-7, 1e-10 gibi diğer düşük değerler modelin performansına genelde bir etki etmemektedir ancak büyük değerler(1, 10 gibi) modelin performansını önemli ölçüde düşürebilir.

Kullanımına dahil diabetes.csv veriseti üzerinde bir örnek [example.py](https://github.com/baranbingol1/naive_bayes_from_scratch/blob/main/example.py) dosyasında verilmiştir.

Dosyaların açıklamaları :

bayes_utils.py : Algoritma için gerekli olan bazı gerekli fonksiyonlar bulunur.\
naive_bayes.py : Algoritmanın ana implemantasyonunu(predict fonksiyonu) ve veri seti üzerinde kullanmak için high-level iki fonksiyon içerir(train ve eval).\
diabetes.csv : örnek veri seti.\
example.py : diabetes.csv verisi üzerinde örnek kullanım.
