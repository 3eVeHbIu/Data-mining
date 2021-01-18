# Практические работы №3-4

## ***Вариант 18***

## *Выполнил студент ББСО-01-18 Арефьев Сергей*

### Импорт необходимых библиотек
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.decomposition import PCA
from sklearn import svm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
### Импорт датасета


```python
headernames = [i for i in range(1, 11)]
headernames.append('Class')
dataset = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
                      names=headernames)
```

### Первичный анализ данных


```python
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.52101</td>
      <td>13.64</td>
      <td>4.49</td>
      <td>1.10</td>
      <td>71.78</td>
      <td>0.06</td>
      <td>8.75</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.51761</td>
      <td>13.89</td>
      <td>3.60</td>
      <td>1.36</td>
      <td>72.73</td>
      <td>0.48</td>
      <td>7.83</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.51618</td>
      <td>13.53</td>
      <td>3.55</td>
      <td>1.54</td>
      <td>72.99</td>
      <td>0.39</td>
      <td>7.78</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.51766</td>
      <td>13.21</td>
      <td>3.69</td>
      <td>1.29</td>
      <td>72.61</td>
      <td>0.57</td>
      <td>8.22</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1.51742</td>
      <td>13.27</td>
      <td>3.62</td>
      <td>1.24</td>
      <td>73.08</td>
      <td>0.55</td>
      <td>8.07</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>209</th>
      <td>210</td>
      <td>1.51623</td>
      <td>14.14</td>
      <td>0.00</td>
      <td>2.88</td>
      <td>72.61</td>
      <td>0.08</td>
      <td>9.18</td>
      <td>1.06</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>210</th>
      <td>211</td>
      <td>1.51685</td>
      <td>14.92</td>
      <td>0.00</td>
      <td>1.99</td>
      <td>73.06</td>
      <td>0.00</td>
      <td>8.40</td>
      <td>1.59</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>211</th>
      <td>212</td>
      <td>1.52065</td>
      <td>14.36</td>
      <td>0.00</td>
      <td>2.02</td>
      <td>73.42</td>
      <td>0.00</td>
      <td>8.44</td>
      <td>1.64</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>212</th>
      <td>213</td>
      <td>1.51651</td>
      <td>14.38</td>
      <td>0.00</td>
      <td>1.94</td>
      <td>73.61</td>
      <td>0.00</td>
      <td>8.48</td>
      <td>1.57</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>213</th>
      <td>214</td>
      <td>1.51711</td>
      <td>14.23</td>
      <td>0.00</td>
      <td>2.08</td>
      <td>73.36</td>
      <td>0.00</td>
      <td>8.62</td>
      <td>1.67</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>214 rows × 11 columns</p>
</div>




```python
dataset['Class'].value_counts()
```




    2    76
    1    70
    7    29
    3    17
    5    13
    6     9
    Name: Class, dtype: int64



Как видно выше, в нашем датасете отсутсвуют данные о классе 4 поэтому, в дальнейшем мы не будем его учитывать.


```python
ps = pd.Series([dataset.loc[dataset['Class'] == i].Class.count()
                    for i in range(1, 8) if i != 4 ],
               index=['1','2','3','5','6','7'])
ps.plot.pie(figsize=(5, 5),label='Classes')
```




    <AxesSubplot:ylabel='Classes'>




![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_8_1.png)


Видно что распределение классов в датасете неравномерное.

### Подготовка данных для обучения модели
Отделяем параметры от значений классов


```python
X = dataset.iloc[:, :-1].values # Параметры
y = dataset.iloc[:, -1].values # Значения классов
```

Далее необходимо разбить выборку на тренировочную и тестовую в соотношении 60/40


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
```

Регулируем масштаб значений параметров так, чтобы каждый параметр имел одинаковый вес.


```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

## SVM-АЛГОРИТМ

![image](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/Support-Vector-Machine.png)

### Обучение модели
Для обучения будем использовать линейное ядро и параметр регуляризации 1.


```python
clf = svm.SVC(C=1, kernel='linear')
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```




    0.9069767441860465



### Улучшение результата.
В данном случае постараемся улучшить результат путем поиска оптимального ядра и параметра регуляризации. Поиск будем проводить при помощи кросс-валидации.

![image](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/cross-validation.svg)

Пример работы кросс-валидации можно увидеть на картинке выше.


```python
gscv = svm.SVC()
param = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
         'C': [i/10 for i in range(1, 16)]}
gscv =  GridSearchCV(clf, param, cv=3, n_jobs=-1, verbose=1)
gscv.fit(X_train, y_train)
```

    Fitting 3 folds for each of 60 candidates, totalling 180 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done 166 tasks      | elapsed:    1.1s
    [Parallel(n_jobs=-1)]: Done 180 out of 180 | elapsed:    1.1s finished





    GridSearchCV(cv=3, estimator=SVC(C=1, kernel='linear'), n_jobs=-1,
                 param_grid={'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                   1.1, 1.2, 1.3, 1.4, 1.5],
                             'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                 verbose=1)



**Наилучшие параметры**


```python
gscv.best_params_
```




    {'C': 0.8, 'kernel': 'linear'}



### Итоговая оценка результата обучения


```python
best_estimator = gscv.best_estimator_
y_pred = best_estimator.predict(X_test)
print('Confusion Matrix:', confusion_matrix(y_test, y_pred), '\n',
      'Отчет о классификации:', classification_report(y_test, y_pred),
      f'Точность предсказания: {accuracy_score(y_test,y_pred)}', sep='\n')
```

    Confusion Matrix:
    [[29  1  0  0  0  0]
     [ 0 25  0  0  0  0]
     [ 0  3  3  0  0  0]
     [ 0  0  0  4  0  1]
     [ 0  0  0  1  2  0]
     [ 0  0  2  0  0 15]]
    
    
    Отчет о классификации:
                  precision    recall  f1-score   support
    
               1       1.00      0.97      0.98        30
               2       0.86      1.00      0.93        25
               3       0.60      0.50      0.55         6
               5       0.80      0.80      0.80         5
               6       1.00      0.67      0.80         3
               7       0.94      0.88      0.91        17
    
        accuracy                           0.91        86
       macro avg       0.87      0.80      0.83        86
    weighted avg       0.91      0.91      0.90        86
    
    Точность предсказания: 0.9069767441860465


**Итоговая точность: 90,69%**

### Визуализация SVM

Ниже предоставлены вспомогательные функции для построения поверхности принятия решения. В функции model_to_2d используется метод главных компонент, для того чтобы мы могли показать работу в двумерной плоскости. Данные функции будут использоваться и для KNN. 


```python
def visualisate_model(pca_2d, plt):
    for i in range(0, pca_2d.shape[0]):
        if y_train[i] == 1:
            c1 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='r', s=50, marker='+')
        elif y_train[i] == 2:
            c2 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='g',s=50, marker='o')
        elif y_train[i] == 3:
            c3 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='b',s=50, marker='*')
        elif y_train[i] == 5:
            c5 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='orange', s=50, marker='+')
        elif y_train[i] == 6:
            c6 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='purple', s=50, marker='o')
        elif y_train[i] == 7:
            c7 = plt.scatter(pca_2d[i,0], pca_2d[i,1], c='yellow', s=50, marker='*')
    plt.legend([c1, c2, c3, c5, c6, c7], ['class 1', 'class 2', 'class 3', 'class 5', 'class 6', 'class 7'])
    
    
def model_to_2d():
    pca = PCA(n_components=2).fit(X_train)
    pca_2d = pca.transform(X_train)
    x_min, x_max = pca_2d[:, 0].min() - 1,   pca_2d[:,0].max() + 1
    y_min, y_max = pca_2d[:, 1].min() - 1,   pca_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
    return pca_2d, xx, yy
```

Для начала построим зависимость точности предсказания от значения С.


```python
C = [i/10 for i in range(1, 17)]
y = []
for i in C:
    clf = svm.SVC(C=i, kernel='linear')
    clf.fit(X_train, y_train)
    y.append(clf.score(X_test, y_test))
plt.plot(C, y, color = 'red', linestyle = 'solid')
plt.title('Зависимость точности предсказания от значения С')
plt.show()
```


![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_28_0.png)


Ниже представлена функция для визуализации самого метода опорных векторов.


```python
def visualisate_svm(kernel, C):
    pca_2d, xx, yy = model_to_2d()
    svmClassifier_2d = svm.SVC(C=C, kernel=kernel).fit(pca_2d, y_train)
    Z = svmClassifier_2d.predict(np.c_[xx.ravel(),  yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)
    plt.title(f'Поверхность принятия решения метода опорных векторов с использованием ядра - {kernel}')
    plt.axis('off')
    visualisate_model(pca_2d, plt)
    plt.show()
```

Посмотрим как будет работать алгоритм с разными ядрами (для C будем использовать лучший параметр найденый при помощи GridSearchCV)


```python
C = gscv.best_params_['C']
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:    
    visualisate_svm(kernel=kernel, C=C)
```


![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_32_0.png)



![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_32_1.png)



![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_32_2.png)



![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_32_3.png)




## KNN-АЛГОРИТМ

![image](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/knn.png)

### Обучение модели
В этот раз для обучения используем алгоритм k-d дерева, колличество соседей равное 3, а также значение p равное 3.


```python
knn = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', p=3)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
```




    0.872093023255814



**Улучшение результата.**
Так же используем кросс-валидацию для поиска наилучших параметров.


```python
knn = KNeighborsClassifier()
param = {'n_neighbors': [i for i in range(1, 5)],
         'weights': ['uniform', 'distance'],
         'algorithm': ['ball_tree', 'kd_tree', 'brute'], 
         'p': [i for i in range(1,6)]}

gknn =  GridSearchCV(knn, param, cv=3, verbose=1)
gknn.fit(X_train, y_train)
```

    Fitting 3 folds for each of 120 candidates, totalling 360 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 360 out of 360 | elapsed:    1.1s finished





    GridSearchCV(cv=3, estimator=KNeighborsClassifier(),
                 param_grid={'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                             'n_neighbors': [1, 2, 3, 4], 'p': [1, 2, 3, 4, 5],
                             'weights': ['uniform', 'distance']},
                 verbose=1)



**Наилучшие параметры**


```python
gknn.best_params_
```




    {'algorithm': 'ball_tree', 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}




```python
best_estimator = gknn.best_estimator_
y_pred = best_estimator.predict(X_test)
print('Confusion Matrix:', confusion_matrix(y_test, y_pred),
      "Отчет о классификации:", classification_report(y_test, y_pred),
      f'Точность предсказания: {accuracy_score(y_test,y_pred)}', sep='\n')
```

    Confusion Matrix:
    [[29  1  0  0  0  0]
     [ 0 25  0  0  0  0]
     [ 1  1  4  0  0  0]
     [ 0  2  0  3  0  0]
     [ 0  0  0  0  3  0]
     [ 0  1  0  0  1 15]]
    Отчет о классификации:
                  precision    recall  f1-score   support
    
               1       0.97      0.97      0.97        30
               2       0.83      1.00      0.91        25
               3       1.00      0.67      0.80         6
               5       1.00      0.60      0.75         5
               6       0.75      1.00      0.86         3
               7       1.00      0.88      0.94        17
    
        accuracy                           0.92        86
       macro avg       0.92      0.85      0.87        86
    weighted avg       0.93      0.92      0.92        86
    
    Точность предсказания: 0.9186046511627907


**Итоговая точность: 91,86%**

### Визуализация KNN

Для начала построим график зависимости точности предсказания от значения колличества соседей (n_neighbors).


```python
n = [i for i in range(2, 20)]
y = []
for i in n:
    knn = KNeighborsClassifier(n_neighbors=i, algorithm='ball_tree', p=1)
    knn.fit(X_train, y_train)
    y.append(knn.score(X_test, y_test))
plt.plot(n, y, color = 'red', linestyle = 'solid')
plt.title('Зависимость точности предсказания от значения колличества соседей')
plt.show()
```


![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_44_0.png)


Далее построим график зависимости точности предсказания от значения p


```python
p = [i for i in range(1,6)]
y = []
for i in p:
    knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', p=i)
    knn.fit(X_train, y_train)
    y.append(knn.score(X_test, y_test))
plt.plot(p, y, color = 'red', linestyle = 'solid')
plt.title('Зависимость точности предсказания от значения p')
plt.show()
```


![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_46_0.png)


Ниже предоставлена функция для визуализации метода k-ближайших соседей.


```python
def visualisate_knn(n_neighbors=5, h = .02, weights = 'distance', algorithm='ball_tree', p=1):
    pca_2d, xx, yy = model_to_2d()
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'white', '#B0F03D', 'grey', '#ffbdd3'])
    knnClassifier_2d = KNeighborsClassifier(n_neighbors, 
                                            weights=weights,
                                            algorithm=algorithm,
                                            p=p).fit(pca_2d, y_train)
    Z = knnClassifier_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'''Поверхность принятия решений метода k-ближайших соседей
    С колличеством соседей - {n_neighbors}
    Алгоритмом - {algorithm}
    Значением weights - {weights}''')
    visualisate_model(pca_2d, plt)
    plt.show()
```

Посмотрим как будет работать данный метод с использованием различных значений weights (остальные параметры примем как лучшие параметры найденые при помощи GridSearchCV).
Также ниже для сравнения выведем поверхность принятия решения при значениях n_neighbors 2 и 9.


```python
n_neighbors = gknn.best_params_['n_neighbors']
p = gknn.best_params_['p']
algorithm = gknn.best_params_['algorithm']
for weights in  ['uniform', 'distance']: 
    visualisate_knn(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p)
weights = gknn.best_params_['weights']
for n_neighbors in  [2, 9]:
    visualisate_knn(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p)
```


![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_50_0.png)



![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_50_1.png)



![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_50_2.png)



![png](https://raw.githubusercontent.com/sergo2048/Data-mining/main/SVM%20KNN/images/output_50_3.png)


### Заключение
По итогу проделанной работы можно заметить, что точность работы метода k-ближайших соседей оказалось больше, чем точность метода опорных векторов. Однако, как мне кажется, данное заявление нельзя считать полностью достоверным, так как обучение производилось на сравнительно не большой выборке данных. 

