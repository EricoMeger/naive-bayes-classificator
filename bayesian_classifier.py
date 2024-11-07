import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath, delimiter=',', encoding="latin-1")
    
    def clean_data(self):
        self.data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        self.data.rename(columns={'v1': 'rotulo', 'v2': 'texto'}, inplace=True)
        self.data.drop_duplicates(inplace=True)

    #facilitar a vida
    @staticmethod
    def process_text(texto):
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        texto = texto.lower()
        return texto.split()

    def apply_preprocessing(self):
        self.data['texto_processado'] = self.data['texto'].apply(self.process_text)

class BayesianClassifier:
    def __init__(self, data):
        self.train_data, self.test_data = train_test_split(data, test_size=0.3, random_state=42)
        self.train_data.reset_index(drop=True, inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)
        self.frequencia_spam = {}
        self.frequencia_ham = {}
        self.prob_spam = {}
        self.prob_ham = {}
        self.lista_palavras = []

    def calcular_frequencias(self):
        for _, row in self.train_data.iterrows():
            for palavra in row['texto_processado']:
                if palavra not in self.lista_palavras:
                    self.lista_palavras.append(palavra)

                if row['rotulo'] == 'spam':
                    if palavra in self.frequencia_spam:
                        self.frequencia_spam[palavra] += 1
                    else:
                        self.frequencia_spam[palavra] = 1
                else:
                    if palavra in self.frequencia_ham:
                        self.frequencia_ham[palavra] += 1
                    else:
                        self.frequencia_ham[palavra] = 1

    def calcular_probabilidades(self):
        """Calcula as probabilidades sem suavização."""
        total_spam = sum(self.frequencia_spam.values())
        total_ham = sum(self.frequencia_ham.values())
        
        for palavra in self.lista_palavras:
            if palavra in self.frequencia_spam:
                self.prob_spam[palavra] = self.frequencia_spam[palavra] / total_spam
            else:
                self.prob_spam[palavra] = 1e-5  # probabiiliadde baixa ao inves de definir 0

            if palavra in self.frequencia_ham:
                self.prob_ham[palavra] = self.frequencia_ham[palavra] / total_ham
            else:
                self.prob_ham[palavra] = 1e-5 

    def classificar(self):
        prob_spam_total = self.train_data['rotulo'].value_counts()['spam'] / len(self.train_data) if 'spam' in self.train_data['rotulo'].value_counts() else 0
        prob_ham_total = self.train_data['rotulo'].value_counts()['ham'] / len(self.train_data) if 'ham' in self.train_data['rotulo'].value_counts() else 0
        
        lista_predicoes = []
        for _, row in self.test_data.iterrows():
            probabilidade_spam = prob_spam_total
            probabilidade_ham = prob_ham_total

            for palavra in row['texto_processado']:
                if palavra in self.prob_spam:
                    probabilidade_spam *= self.prob_spam[palavra]
                else:
                    probabilidade_spam *= 1e-5
                if palavra in self.prob_ham:
                    probabilidade_ham *= self.prob_ham[palavra]
                else:
                    probabilidade_ham *= 1e-5

            if probabilidade_spam >= probabilidade_ham:
                lista_predicoes.append('spam')
            else:
                lista_predicoes.append('ham')

        return lista_predicoes

    def calcular_acuracia(self, predicoes):
        acertos = 0
        for i in range(len(self.test_data)):
            if predicoes[i] == self.test_data['rotulo'][i]:
                acertos += 1

        return (acertos / len(self.test_data)) * 100


# ******************* EXECUÇÃO DO CLASSIFICADOR ****************************************************

preprocessor = Preprocessor(filepath='./spam.csv')
preprocessor.clean_data()
preprocessor.apply_preprocessing()

classificador = BayesianClassifier(preprocessor.data)
classificador.calcular_frequencias()
classificador.calcular_probabilidades()
predicoes = classificador.classificar()
acuracia = classificador.calcular_acuracia(predicoes)

print(f"Acurácia: {acuracia:.2f}%")
