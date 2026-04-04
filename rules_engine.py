import pandas as pd
import json
from itertools import product
from bayes_alerts import BayesianNetwork
from sklearn.metrics import classification_report, accuracy_score

def load_rules(path="regras.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["regras"]

def check_condition(value, operador, threshold):
    ops = {
        ">": lambda x: x > threshold,
        "<": lambda x: x < threshold,
        ">=": lambda x: x >= threshold,
        "<=": lambda x: x <= threshold,
        "==": lambda x: x == threshold,
    }
    return ops.get(operador, lambda x: False)(value)

def aplicar_regras(df, regras):
    def avaliar(row):
        for regra in regras:  # prioridade pela ordem
            if all(
                check_condition(row[var], cond["operador"], cond["valor"])
                for var, cond in regra["condicoes"].items()
            ):
                return regra["alerta"]
        return "indefinido"

    df = df.copy()
    df["estado"] = df.apply(avaliar, axis=1)
    return df

def discretizar(df):
    d = pd.DataFrame()

    d['temp'] = pd.cut(
        df['temperature_c'],
        bins=[-10, 20, 30, 35, 40, 60],
        labels=['frio', 'ameno', 'quente', 'muito_quente', 'extremo'],
        include_lowest=True
    )

    d['hum'] = pd.cut(
        df['humidity_percent'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['muito_baixa', 'baixa', 'media', 'alta', 'muito_alta'],
        include_lowest=True
    )

    d['vento'] = pd.cut(
        df['wind_speed_kmh'],
        bins=[0, 20, 40, 60, 120],
        labels=['fraco', 'moderado', 'forte', 'muito_forte'],
        include_lowest=True
    )

    d['estado'] = df['estado']

    return d.dropna()

def main():

    # Load dataset
    df = pd.read_csv("data/processed_lisboa_porto_air_quality.csv", delimiter=";")

    # # Cleaning
    df = df[df.city != "UCI_Dataset"]
    df = df.drop(columns=["C6H6", "NMHC", "NOx"], errors="ignore")

    # # Load rules
    regras = load_rules("regras.json")

    # Apply rules
    df = aplicar_regras(df, regras)

    print("\nDistribuição do estado:")
    print(df["estado"].value_counts())

    # # Save
    df.to_csv("data/processed_lisboa_porto_air_quality_filtered.csv", index=False)

    # Discretização
    df_disc = discretizar(df)

    # Train BN
    bn = BayesianNetwork()
    bn.fit(df_disc)

    # Test query
    probs = bn.query_estado(temp='extremo', hum='muito_baixa', vento='muito_forte')
    print("\nExemplo de inferência:")
    print(probs)

    # Avaliação
    resultados = []

    for _, row in df_disc.iterrows():
        probs = bn.query_estado(
            temp=row['temp'],
            hum=row['hum'],
            vento=row['vento']
        )

        melhor = max(probs, key=probs.get)

        resultados.append({
            'real': row['estado'],
            'pred': melhor
        })

    df_bn = pd.DataFrame(resultados)

    acc = accuracy_score(df_bn['real'], df_bn['pred'])
    print(f"\nAccuracy: {acc:.2%}")

    print("\nClassification Report:")
    print(classification_report(df_bn['real'], df_bn['pred']))


if __name__ == "__main__":
    main()
