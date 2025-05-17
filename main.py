from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

column_names = [
    'Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
    'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'
]

# Citire date din fisierul 'abalone.data'
data = pd.read_csv('abalone.data', header=None, names=column_names)

# Separare caracteristici (X) si eticheta (y)
X = data.drop('Rings', axis=1)
y = data['Rings']

# Conversie coloana 'Sex' din categorica în numerica
encoder = LabelEncoder()
X['Sex'] = encoder.fit_transform(X['Sex'])

# Scalare date -> aceeași scară
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Creare lista cu caracteristici pt grafice
features = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']

# Split la date in set de antrenare si set de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)
print(f"Numărul de rânduri pentru setul de antrenament: {len(X_train)}")
print(f"Numărul de rânduri pentru setul de test: {len(X_test)}")


# ---------------Definire retele neurale ---------------

# Seturi de parametri pentru grid search
hidden_layer_options = [1, 2]  # Numar de straturi ascunse (1 sau 2)
neuron_options = ['same', 'half', 'double']  # Nr. neuroni fata de stratul anterior
learning_rate_options = [0.1, 0.01]  # Learning rate

# Creare combinatii posibile pentru parametrii de mai sus
param_combinations = list(product(hidden_layer_options, neuron_options, learning_rate_options))

# --------------- Testarea tuturor combinatiilor de parametri ---------------

best_mse = float('inf')  # Initializare eroarea medie patratica cu o valoare mare
best_r2 = -float('inf')  # Initializăm R² cu o valoare mica
best_params = None  # Parametrii optimi

results = []  # Lista pentru salvarea tuturor rezultatelor/combinatiilor posibile

# Iterare prin toate combinatiile de parametri
for hidden_layers, neuron_option, lr in param_combinations:
    # Stabilire nr de neuroni pentru fiecare strat ascuns
    if neuron_option == 'same':
        neurons = [32] * hidden_layers
    elif neuron_option == 'half':
        neurons = [32, 16] if hidden_layers == 2 else [32]
    elif neuron_option == 'double':
        neurons = [32, 64] if hidden_layers == 2 else [32]

    # Initializare model MLPRegressor cu parametrii curenti
    model = MLPRegressor(
        hidden_layer_sizes=tuple(neurons),  # Setare dimensiuni straturi ascunse
        activation='tanh',  # Functia de activare tangenta hiperbolica/ ReLU
        solver='adam',  # Algoritmul de optimizare ( Adam )
        learning_rate_init=lr,  # Rata de invatare
        max_iter=1000,  # Nr maxim de cicluri de antrenare
        random_state=60,  # Rezultate identice intre rulari -> debugging eficient
        alpha=0.001,  # Previnire overfitting
        early_stopping=True,  # Oprire pentru prevenire overfitting
        n_iter_no_change=10,  # Oprire după 10 cicluri fara schimbari semnificative
        tol=0.001,  # Determinare toleranta minima necesara pentru oprire
        epsilon=0.005  # prevenire impartire la zero
    )

    # Antrenare model pe datele de train
    model.fit(X_train, y_train)

    # Predictie valori pentru setul de test
    y_pred = model.predict(X_test)

    # Calculare eroarea pătratică medie (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Calculare R² Score pentru a evalua performanta modelului (procent)
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100  # Conversie in procente

    results.append({
        'Config': f'{hidden_layers} layers, {neuron_option}, LR={lr}',
        'Accuracy (%)': accuracy
    })

    # Comparare rezultate si pastratea celui mai bun set de parametri
    if mse <= best_mse:
        best_mse = mse
        best_r2 = r2
        best_params = (hidden_layers, neuron_option, lr)

# --------------- Afisarea celor mai bune rezultate ---------------

print(f"\nCele mai bune rezultate sunt cu parametrii: ")
print(f"Numărul de straturi ascunse: {best_params[0]}")
print(f"Tipul de neuroni pe straturi: {best_params[1]}")
print(f"Learning rate: {best_params[2]}")
print(f"\nMean Squared Error (MSE): {best_mse:.2f}")
print(f"Acuratețea modelului (R² Score): {best_r2 * 100:.2f}%")

#Afisare grafice de tip scatter: nr de rings functie de fiecare caracteristica fizica in parte
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)  # 3x3 subgrafice
    plt.scatter(data[feature], data['Rings'], alpha=0.5)
    plt.title(f'Rings vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Rings')

plt.tight_layout()
plt.show()


# Plot predictie vs valori reale
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Valori Reale')
plt.ylabel('Predicții')
plt.title('Predicție vs. Valori Reale')
plt.tight_layout()
plt.show()

# ------------------Afisarea acuratetei in functie de toate combinatiile posibile ale hiperparametrilor---------------------

# 1. Convertim lista într-un DataFrame și sortăm
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy (%)')

# 2. Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(results_df['Config'], results_df['Accuracy (%)'], color='skyblue')
plt.xlabel('R² Score (%)')
plt.title('Performanța pentru fiecare combinație de hiperparametri')
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f'{width:.2f}%',
        va='center',
        fontsize=9,
        color='black'
    )

plt.show()