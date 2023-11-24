from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Carregando dados do Titanic
#url = 'https://learnenough.s3.amazonaws.com/titanic.csv'
url = 'C:/Users/aninh/Downloads/titanic.csv'
titanic = pd.read_csv(url, index_col='Name')

# Preprocessamento básico removendo colunas irrelevantes e preenchendo valores ausentes.
titanic = titanic.drop(['Cabin', 'Ticket'], axis=1)
titanic = titanic.dropna()

# Criando variáveis categóricas
titanic = pd.get_dummies(titanic, columns=['Sex', 'Embarked'], drop_first=True)

# Separando features (X) e labels (y)
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# dividindo d5ados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Função para treinar e avaliar o modelo
def train_and_evaluate_model(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    confusion_mat = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, confusion_mat

# Rota principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Receba os parâmetros do formulário
        classifier_name = request.form.get('classifier')
        param1 = float(request.form.get('param1'))
        param2_str = request.form.get('param2')

        # Verifique se param2_str é uma string não vazia antes de converter para float
        param2 = float(param2_str) if param2_str else None

        # Crie o classificador com os parâmetros escolhidos
        if classifier_name == 'KNN':
            classifier = KNeighborsClassifier(n_neighbors=int(param1))
        elif classifier_name == 'SVM':
            classifier = SVC(C=param1, kernel='linear')
        elif classifier_name == 'MLP':
            classifier = MLPClassifier(hidden_layer_sizes=(int(param1),), max_iter=int(param2))
        elif classifier_name == 'DT':
            classifier = DecisionTreeClassifier(max_depth=int(param1))
        elif classifier_name == 'RF':
            classifier = RandomForestClassifier(n_estimators=int(param1))

        # Treine e avalie o modelo
        accuracy, precision, recall, f1, confusion_mat = train_and_evaluate_model(classifier, X_train, y_train, X_test, y_test)

        # Gere a imagem da matriz de confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # Salve a imagem em um buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Converta a imagem para base64
        conf_mat_image = base64.b64encode(buf.read()).decode('utf-8')

        return render_template('result.html', accuracy="%.2f%%" % (accuracy * 100.0), precision="%.2f%%" % (precision * 100.0), recall="%.2f%%" % (recall * 100.0), f1="%.2f%%" % (f1 * 100.0), conf_mat_image=conf_mat_image)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)