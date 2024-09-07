import streamlit as st # biblioteca que permite criar aplicativos web com python.
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("pizzas.csv")
modelo = LinearRegression()

# as colunas da minha base de dados
x = df[["diametro"]]
y = df[["preco"]]

# treinando ia com os dados
modelo.fit(x, y)

# CRIANDO A APLICAÇÃO WEB COM A BIBLIOTECA DO stremalit
st.title("Prevendo o valor de uma pizza")
st.divider()

diametro = st.number_input("Digite o tamanho do diametro da pizza em centímetros: ")

if diametro != 0:
    preco_previsto = modelo.predict([[diametro]])[0][0]
    st.write(f"O valor da pizza com o diametro de {diametro:.2f} cm é de R${preco_previsto:.2f}.")
    st.balloons()

st.divider()
st.subheader("Gráfico de Dispersão")
st.markdown("""
            O gráfico abaixo é um gráfico de disperção que está mostrando a correlação entre o preço e o diametro da pizza dado em centímetros (cm). Conforme o diametro do pedido almenta, o proço da pizza tbm irá almentar.
        """)

fig, ax = plt.subplots()
df.plot(kind="scatter", x="diametro", y="preco", ax=ax)
ax.set_title("Diâmetro vs Preço")
ax.set_xlabel("Diâmetro (cm)")
ax.set_ylabel("Preço (R$)")
st.pyplot(fig)

# comando de ativação no terminal: streamlit run app.py