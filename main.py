import pickle

import pydotplus
from category_encoders import OrdinalEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from yellowbrick.classifier import ConfusionMatrix


# Função para dividir os dados e processá-los
def split_data_with_processing(dataframe):
    # Removendo a coluna "Exemplo"
    dataframe = dataframe.drop(columns=["Exemplo"])

    # Inicializando os codificadores para as características
    le_alternativo = LabelEncoder()
    le_bar = LabelEncoder()
    le_sex_sab = LabelEncoder()
    le_fome = LabelEncoder()
    le_chuva = LabelEncoder()
    le_res = LabelEncoder()
    le_conc = LabelEncoder()

    # Definindo mapeamentos ordinais para algumas colunas
    client_mapping = [{'col': 'Cliente', 'mapping': {'Nenhum': 0, 'Alguns': 1, 'Cheio': 2}},
                      {'col': 'Preço', 'mapping': {'R': 0, 'RR': 1, 'RRR': 2}},
                      {'col': 'Tempo', 'mapping': {'0-10': 0, 'out/30': 1, '30-60': 2, '>60': 3, }}]
    oe_client = OrdinalEncoder(mapping=client_mapping)

    # Convertendo características categóricas em numéricas
    dataframe["Alternativo"] = le_alternativo.fit_transform(dataframe["Alternativo"])
    dataframe["Bar"] = le_bar.fit_transform(dataframe["Bar"])
    dataframe["Sex/Sab"] = le_sex_sab.fit_transform(dataframe["Sex/Sab"])
    dataframe["fome"] = le_fome.fit_transform(dataframe["fome"])
    dataframe["Chuva"] = le_chuva.fit_transform(dataframe["Chuva"])
    dataframe["Res"] = le_res.fit_transform(dataframe["Res"])
    dataframe["conc"] = le_conc.fit_transform(dataframe["conc"])

    # Aplicando mapeamentos ordinais
    dataframe = oe_client.fit_transform(dataframe)

    # One-hot encoding para a coluna "Tipo"
    dataframe = pd.get_dummies(dataframe, columns=["Tipo", ])

    print(dataframe.columns)
    print(dataframe.values)

    # Dividindo os dados em treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=["conc", ]),
        dataframe["conc"],
        test_size=0.2,
        random_state=20
    )

    # Salvando os conjuntos de dados em um arquivo usando pickle
    with open("restaurante_train_test_with.pkl", "wb") as f:
        pickle.dump((x_train.columns.to_list(), x_train.values, x_test.values, y_train.values, y_test.values), f)


# Função para treinar e visualizar uma árvore de decisão com dados processados
def plot_tree_with_processing():
    print("Treinando Arvore Com Processamento")

    # Lendo os dados
    df = pd.read_csv("restaurantev2.csv", delimiter=";", encoding="ISO-8859-1")
    split_data_with_processing(df)

    # Carregando os conjuntos de dados treino e teste do arquivo usando o pickle
    with open("restaurante_train_test_with.pkl", 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

    # Inicializando e treinando a árvore de decisão
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train, y_train)

    # Predizendo os resultados para o conjunto de teste
    y_pred = model.predict(x_test)
    print(y_pred)

    # Calculando métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Acuracia: {accuracy}", )
    print(f"Matriz de confusao: {confusion}", )
    print(report)

    # Visualizando a matriz de confusão
    cm = ConfusionMatrix(model, classes=["Nao", "Sim"])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(outpath=f"{model.__class__.__name__}_confusion_matrix_with.png")

    plt.close()

    # Visualizando a árvore de decisão
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=["Nao", "Sim"],
        filled=True,
        rounded=True
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(f"{model.__class__.__name__}_with_graph.png")


# Função para dividir os dados sem processamento adicional (sem mapeamento ordinal)
def split_data_without_processing(dataframe):
    # Removendo a coluna "Exemplo"
    dataframe = dataframe.drop(columns=["Exemplo"])

    # Inicializando os codificadores para as características
    le_alternativo = LabelEncoder()
    le_bar = LabelEncoder()
    le_sex_sab = LabelEncoder()
    le_fome = LabelEncoder()
    le_chuva = LabelEncoder()
    le_res = LabelEncoder()
    le_conc = LabelEncoder()

    # Convertendo características categóricas em numéricas
    dataframe["Alternativo"] = le_alternativo.fit_transform(dataframe["Alternativo"])
    dataframe["Bar"] = le_bar.fit_transform(dataframe["Bar"])
    dataframe["Sex/Sab"] = le_sex_sab.fit_transform(dataframe["Sex/Sab"])
    dataframe["fome"] = le_fome.fit_transform(dataframe["fome"])
    dataframe["Chuva"] = le_chuva.fit_transform(dataframe["Chuva"])
    dataframe["Res"] = le_res.fit_transform(dataframe["Res"])
    dataframe["conc"] = le_conc.fit_transform(dataframe["conc"])

    # One-hot encoding para a colunas "Tipo", "Tempo", "Preço", "Cliente"
    dataframe = pd.get_dummies(dataframe, columns=["Tipo", "Tempo", "Preço", "Cliente"])

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=["conc", ]),
        dataframe["conc"],
        test_size=0.2,
        random_state=20
    )

    # Salvando os conjuntos de dados em um arquivo pickle
    with open("restaurante_train_test.pkl", "wb") as f:
        pickle.dump((x_train.columns.to_list(), x_train.values, x_test.values, y_train.values, y_test.values), f)


# Função para treinar e visualizar uma árvore de decisão sem processamento adicional
def plot_tree_without_processing():
    print("Treinando Arvore Sem Processamento")

    # Treinar e visualizar uma árvore de decisão com processamento
    df = pd.read_csv("restaurantev2.csv", delimiter=";", encoding="ISO-8859-1")
    split_data_without_processing(df)

    # Carregando os conjuntos de dados treino e teste do arquivo usando pickle
    with open("restaurante_train_test.pkl", 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

    # Inicializando e treinando a árvore de decisão
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train, y_train)

    # Predizendo os resultados para o conjunto de teste
    y_pred = model.predict(x_test)
    print(y_pred)

    # Calculando métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Acuracia: {accuracy}", )
    print(f"Matriz de confusao: {confusion}", )
    print(report)

    cm = ConfusionMatrix(model, classes=["Nao", "Sim"])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(outpath=f"{model.__class__.__name__}_confusion_matrix_without.png")

    plt.close()

    # Visualizando a árvore de decisão
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=["Nao", "Sim"],
        filled=True,
        rounded=True
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(f"{model.__class__.__name__}_graph.png")


if __name__ == '__main__':
    # Treinar e visualizar uma árvore de decisão sem processamento adicional
    plot_tree_without_processing()
    # Treinar e visualizar uma árvore de decisão com processamento
    plot_tree_with_processing()
