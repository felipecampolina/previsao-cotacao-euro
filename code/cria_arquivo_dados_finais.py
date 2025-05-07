import pandas as pd
from functools import reduce

# 1) Ler os arquivos ---------------------------------------------------------
sofr = pd.read_csv("../data/sofr.csv", sep=';')
curacy = pd.read_csv("../data/curency.csv", sep=';')
brent = pd.read_csv("../data/brentOil.csv", sep=';')

# ── 2. Converter a coluna de datas ─────────────────────────────────────────────
for df in (sofr, brent, curacy):
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)   # dd/mm/aaaa ➜ datetime64

# ── 3. Manter só datas comuns aos três ficheiros ───────────────────────────────
common_dates = set(sofr['date']) & set(brent['date']) & set(curacy['date'])

sofr   = sofr  [sofr  ['date'].isin(common_dates)]
brent  = brent [brent ['date'].isin(common_dates)]
curacy = curacy[curacy['date'].isin(common_dates)]

# ── 4. Construir o df consolidado ──────────────────────────────────────────────
df = pd.DataFrame({'Date': sorted(common_dates)})  # começa apenas com as datas

df = (
    df
    .merge(
        sofr[['date', 'rate']].rename(columns={'date': 'Date', 'rate': 'SOFR'}),
        on='Date', how='left'
    )
    .merge(
        brent[['date', 'ultimo']].rename(columns={'date': 'Date', 'ultimo': 'Brent'}),
        on='Date', how='left'
    )
    .merge(
        curacy[['date', 'usEuro', 'yeanEuro']].rename(
            columns={'date': 'Date', 'usEuro': 'USD_EUR', 'yeanEuro': 'JPY_EUR'}
        ),
        on='Date', how='left'
    )
    .sort_values('Date')
    .reset_index(drop=True)
)

# ── 5. Trocar vírgula decimal por ponto e converter para float ────────────────
for col in ['SOFR', 'Brent', 'USD_EUR', 'JPY_EUR']:
    df[col] = (
        df[col]
        .astype(str)        # garante string
        .str.replace(',', '.', regex=False)
        .astype(float)      # para float
    )

# ── 6. [Opcional] Guardar o resultado ─────────────────────────────────────────
df.to_csv("../data/dados_unificados.csv", index=False)
