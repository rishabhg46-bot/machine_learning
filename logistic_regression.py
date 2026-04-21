import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
    else:
        X, y = make_blobs(n_features=2, centers=3, random_state=2)

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
    return X, y

def draw_meshgrid(X):
    a = np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.01)
    b = np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.01)

    XX, YY = np.meshgrid(a, b)
    input_array = np.c_[XX.ravel(), YY.ravel()]

    return XX, YY, input_array

plt.style.use('fivethirtyeight')

st.sidebar.title("Logistic Regression")

dataset = st.sidebar.selectbox('Dataset', ('Binary', 'Multiclass'))
penalty = st.sidebar.selectbox('Penalty', ('l2', 'l1', 'elasticnet', 'none'))
C = st.sidebar.number_input('C', value=1.0)
solver = st.sidebar.selectbox('Solver', ('lbfgs', 'liblinear', 'saga'))
max_iter = st.sidebar.number_input('Max Iter', value=100)

l1_ratio = st.sidebar.number_input('l1_ratio', value=0.5)

fig, ax = plt.subplots()
X, y = load_initial_graph(dataset, ax)

st.pyplot(fig)   # ALWAYS show initial graph

if st.sidebar.button('Run Algorithm'):

    # ✅ FIX: handle invalid combinations
    if penalty == "l1":
        solver = "liblinear"
    elif penalty == "elasticnet":
        solver = "saga"

    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=int(max_iter),
        l1_ratio=l1_ratio if penalty == "elasticnet" else None
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid(X)
    labels = clf.predict(input_array)

    fig2, ax2 = plt.subplots()
    ax2.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')

    st.pyplot(fig2)

    st.subheader(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")