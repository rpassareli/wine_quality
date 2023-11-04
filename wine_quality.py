import numpy as np  # matriz e vetores
import pandas as pd  # ler arquivos
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # graficos
# pega todos os dados tipados como objetos, pega todos os valores diferentes e discretiza os valores 0 a n
from sklearn.preprocessing import LabelEncoder
# divide os dados em teste e treino
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # arvore de decisão
from sklearn.metrics import accuracy_score  # calcula a precisão
# matriz de confusão para verificar quantos erros e quantos acertos foram feitos
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import StandardScaler

# 1º passo
data = pd.read_csv('WineQT.csv')
# print(data.info())  # mostra as informações de cada coluna e verifica se tem valores nulos ou atributos desnecessários (FOI VERIFICADO QUE Id é desnecessário)

# 2º passo
data = data.drop(['Id'], axis=1)  # removendo atributo Id por ser inutil
# print(data.info()) # verificado se o atributo Id foi removido (NESSE CASO Id foi removido)
# print(data['quality'].unique())  # mostrando o que tem dentro da coluna quality [0...10] (Nesse dataframe a qualidade varia em 3,4,5,6,7 e 8)

# 3º passo
le = LabelEncoder()  # instanciando a classe para transformar transformar a qualidade em forma discretizada para o valores 3,4,5,6,7 e 8

# 4º passo
data['quality'] = le.fit_transform(
    data['quality'])  # transformando qualidade de 3,4,5,6,7 e 8 em forma discretizada 0,1,2,3,4,5
# print(data['quality'].unique()) # mostrando a tranformação de quality [5 6 7 4 8 3] => [2 3 4 1 5 0]

# 5º passo
# removendo a coluna quality ou usar a forma abaixo para x transformada em array, ou seja, não fica mais no formato de pandas o dataframe
x = data.drop(['quality'], axis=1).values
# x = data.iloc[:, 1:31].values #removendo a coluna diagnosis de outra forma também
y = data['quality'].values

# 6º passo
# x passa todos os valores com excessão da classe | y passa todos os valores da classe | test_size = 0.3 (30% teste) e train_size = 0.7 (70% treino) e random_state = 0 => vai devolver um array
# x_train é do conjunto treino, x_test é do conjunto teste, y_train é da classe de treino, y_test é da classe de teste

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.10, train_size=0.90, random_state=7152)

scaler = StandardScaler()  # instanciando a classe para normalizar os dados
x_train_scaled = scaler.fit_transform(x_train)  # normalizando os dados
x_test_scaled = scaler.fit_transform(x_test)  # normalizando os dados

# 7º passo => Treinando e testando o modelo com dados não normalizados
clf1 = DecisionTreeClassifier(criterion='entropy', random_state=1)
clf1.fit(x_train, y_train)  # realiza o treinamento
y_pred1 = clf1.predict(x_test)  # realiza a previsão
precision1 = accuracy_score(y_test, y_pred1)  # calcula a precisão
print("Precisão com dados não normalizados: ", precision1)

# 8º passo => Treinando e testando o modelo com dados normalizados
clf2 = DecisionTreeClassifier(criterion='entropy', random_state=1)
clf2.fit(x_train_scaled, y_train)  # realiza o treinamento
y_pred2 = clf2.predict(x_test_scaled)  # realiza a previsão
precision2 = accuracy_score(y_test, y_pred2)  # calcula a precisão
print("Precisão com dados normalizados: ", precision2)

# 9º passo => Matriz de confusão para dados não normalizados
cm1 = confusion_matrix(y_test, y_pred1)  # calcula a matriz de confusão
disp1 = ConfusionMatrixDisplay(cm1, display_labels=clf1.classes_)
# disp1.plot()
# plt.show()

# 10º passo => Matriz de confusão para dados normalizados
cm2 = confusion_matrix(y_test, y_pred2)  # calcula a matriz de confusão
disp2 = ConfusionMatrixDisplay(cm2, display_labels=clf2.classes_)
# disp2.plot()
# plt.show()


knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
knn_acc = accuracy_score(y_test, y_pred_knn)
print("Precisão KNN: ", knn_acc)

# 11º passo
# fig = plt.figure(figsize=(50,25))
# tree.plot_tree(clf, feature_names=data.columns, filled=True,
#                class_names=["0", "1", "2", "3", "4", "5"])
# plt.show()
