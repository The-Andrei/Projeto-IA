import pandas as pd
from itertools import product


class BayesianNetwork:
    """
    Implementação de uma Rede Bayesiana discreta simples.

    Estrutura:
        temp   ─┐
        hum    ─┼──→ estado
        vento  ─┘

    - temp, hum, vento → nós independentes (sem pais)
    - estado → depende de todos os outros
    """

    def __init__(self):
        # CPDs: tabelas de probabilidade condicional
        self.cpds = {}

        # Estrutura do grafo (DAG)
        self.parents = {
            'temp': [],
            'hum': [],
            'vento': [],
            'estado': ['temp', 'hum', 'vento']
        }

        # Lista de categorias possíveis por variável
        self.categories = {}

    def fit(self, df, alpha=1.0):
        """
        Treina a rede:
        - Aprende categorias
        - Calcula probabilidades (CPDs)
        - Usa suavização de Laplace (alpha)
        """

        # ── 1. Guardar categorias de cada variável ──
        for node in self.parents:
            self.categories[node] = list(df[node].astype(str).unique())

        # ── 2. Estimar CPDs ──
        for node, pais in self.parents.items():

            vals_node = self.categories[node]
            self.cpds[node] = {}

            # ── Caso 1: nó sem pais (probabilidade simples) ──
            if not pais:
                counts = df[node].astype(str).value_counts()

                # total com suavização
                total = counts.sum() + alpha * len(vals_node)

                # P(node)
                self.cpds[node][()] = {
                    v: (counts.get(v, 0) + alpha) / total
                    for v in vals_node
                }

            # ── Caso 2: nó com pais (probabilidade condicional) ──
            else:
                vals_pais = [self.categories[p] for p in pais]

                # todas as combinações possíveis dos pais
                for combo in product(*vals_pais):

                    mask = pd.Series([True] * len(df), index=df.index)

                    # filtrar linhas que correspondem à combinação
                    for p, v in zip(pais, combo):
                        mask &= (df[p].astype(str) == v)

                    subset = df[mask][node].astype(str)

                    counts = subset.value_counts()
                    total = counts.sum() + alpha * len(vals_node)

                    # P(node | pais)
                    self.cpds[node][combo] = {
                        v: (counts.get(v, 0) + alpha) / total
                        for v in vals_node
                    }

        print("CPDs treinadas")
        return self

    def p_cond(self, node, value, evidence):
        """
        Retorna:
        P(node = value | evidence)

        - evidence contém valores dos pais
        """

        value = str(value)
        pais = self.parents[node]

        # nó sem pais
        if not pais:
            return self.cpds[node][()].get(value, 1e-9)

        # construir chave com valores dos pais
        key = tuple(str(evidence.get(p, '')) for p in pais)

        return self.cpds[node].get(key, {}).get(value, 1e-9)

    def query_estado(self, temp=None, hum=None, vento=None):
        """
        Inferência:
        Calcula P(estado | evidência)
        """

        evidence = {
            'temp': temp,
            'hum': hum,
            'vento': vento
        }

        probs = {}

        # calcular probabilidade para cada estado possível
        for estado in self.categories['estado']:

            prob = 1.0

            # P(estado | temp, hum, vento)
            prob *= self.p_cond('estado', estado, evidence)

            # multiplicar pelos priors das evidências
            for var in ['temp', 'hum', 'vento']:
                if evidence[var] is not None:
                    prob *= self.p_cond(var, evidence[var], {})

            probs[estado] = prob

        # normalizar probabilidades
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs
