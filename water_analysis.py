import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 1. Carregue o arquivo CSV como um dataframe Pandas
df = pd.read_csv('/home/felipe.treis/Downloads/waterQuality1.csv')

# 2. Faça uma análise prévia sobre o formato dos dados
print("Primeiras 5 linhas do dataframe:")
print(df.head())

print("\nÚltimas 5 linhas do dataframe:")
print(df.tail())

print("\nTipos de dados das colunas:")
print(df.dtypes)

# 3. Lide com erros de leitura e problemas de tipagem
df = df[df['ammonia'].apply(lambda x: x.replace('.', '', 1).isdigit())]
df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')

df = df[df['is_safe'].isin([0, 1])]
df['is_safe'] = pd.to_numeric(df['is_safe'])

# 4. Resolva o problema de desbalanceamento
print("Contagem de amostras por classe antes da reamostragem:")
print(df['is_safe'].value_counts())

# Verifique se há amostras suficientes para reamostragem
if (df['is_safe'] == 0).sum() > 0:
    # Ajuste o número de amostras reamostradas
    num_samples_to_resample = min(912, (df['is_safe'] == 0).sum())
    
    c_0 = df[df['is_safe'] == 0].sample(n=num_samples_to_resample, random_state=1, replace=True)
    c_1 = df[df['is_safe'] == 1]
    df = pd.concat([c_0, c_1])
    df = df.reset_index(drop=True)

    print("\nContagem de amostras por classe após a reamostragem:")
    print(df['is_safe'].value_counts())
else:
    print("Não há amostras suficientes para reamostragem.")

# 5. Faça uma análise exploratória de dados

# 6. Separe os dados em conjuntos de treinamento e teste
if len(df) > 0:
    X = df.drop(columns=['is_safe'])  # Recursos
    y = df['is_safe']  # Rótulos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # 7. Aplique os classificadores
    def apply_classifier(classifier, X_train, y_train, X_test):
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        return predictions

    # Inicialize os classificadores
    nb_classifier = GaussianNB()
    knn_classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    dt_classifier = DecisionTreeClassifier(random_state=1)

    # 8. Avalie o desempenho dos classificadores e imprima o classification report
    for classifier, name in zip([nb_classifier, knn_classifier, dt_classifier], ['Gaussian Naive Bayes', 'K Nearest Neighbors', 'Decision Tree']):
        predictions = apply_classifier(classifier, X_train, y_train, X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, predictions))
else:
    print("Não há amostras disponíveis para análise.")
