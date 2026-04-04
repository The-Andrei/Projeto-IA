import pandas as pd
from itertools import product

class BayesianNetwork:

    def __init__(self):
        self.cpds = {}
        self.parents = {
            'temp': [],
            'hum': [],
            'vento': [],
            'estado': ['temp', 'hum', 'vento']
        }
        self.categories = {}

    def fit(self, df, alpha=1.0):

        # categorias
        for node in self.parents:
            self.categories[node] = list(df[node].astype(str).unique())

        # CPDs
        for node, pais in self.parents.items():
            vals_node = self.categories[node]
            self.cpds[node] = {}

            if not pais:
                counts = df[node].astype(str).value_counts()
                total = counts.sum() + alpha * len(vals_node)

                self.cpds[node][()] = {
                    v: (counts.get(v, 0) + alpha) / total
                    for v in vals_node
                }

            else:
                vals_pais = [self.categories[p] for p in pais]

                for combo in product(*vals_pais):
                    mask = pd.Series([True] * len(df), index=df.index)

                    for p, v in zip(pais, combo):
                        mask &= (df[p].astype(str) == v)

                    subset = df[mask][node].astype(str)
                    counts = subset.value_counts()
                    total = counts.sum() + alpha * len(vals_node)

                    self.cpds[node][combo] = {
                        v: (counts.get(v, 0) + alpha) / total
                        for v in vals_node
                    }

        print("CPDs treinadas")
        return self

    def p_cond(self, node, value, evidence):
        value = str(value)
        pais = self.parents[node]

        if not pais:
            return self.cpds[node][()].get(value, 1e-9)

        key = tuple(str(evidence.get(p, '')) for p in pais)
        return self.cpds[node].get(key, {}).get(value, 1e-9)

    def query_estado(self, temp=None, hum=None, vento=None):

        evidence = {
            'temp': temp,
            'hum': hum,
            'vento': vento
        }

        probs = {}

        for estado in self.categories['estado']:
            prob = 1.0

            # P(estado | pais)
            prob *= self.p_cond('estado', estado, evidence)

            # P(evidência (nós independentes))
            for var in ['temp', 'hum', 'vento']:
                if evidence[var] is not None:
                    prob *= self.p_cond(var, evidence[var], {})

            probs[estado] = prob

        # normalizar
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs