from dotenv import load_dotenv
load_dotenv(dotenv_path='config/secrets.env')

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pysass_crypto as sass_crypto
import logging
from sqlalchemy import create_engine
import os

start_time = time.time()

############################ CONFIGURAÇÕES #################################################################################
############################ CONFIGURAÇÕES DE SEGURANÇA #######################################################
exchange = ccxt.hyperliquid({
    "walletAddress": os.environ['WALLET_ADDRESS'],
    "privateKey": os.environ['PRIVATE_KEY'],
})

# Configuração do logger
log_filename = f"logs/trading_log_{datetime.today().strftime('%Y-%m-%d')}.log"
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Conexão com o banco PostgreSQL (RDS)
DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASS']
DB_HOST = os.environ['DB_HOST']
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ['DB_NAME']

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

###########################################################################################################
# Subtrair 500 dias
data_inicial = (datetime.today() - timedelta(days=500)).strftime('%Y-%m-%d')
query = f"""
    SELECT * FROM ohlcv WHERE date >= '{data_inicial}'
"""
data = pd.read_sql_query(query, engine)

###########################################################################################################

volume = sass_crypto.format_from_database(data[['date', 'symbol', 'volume']])

price = sass_crypto.format_from_database(data[['date', 'symbol', 'close']])

volume = volume * price

###################################### PUXAR OS PREÇOS DA EXCHANGE ####################################



# Obtem a lista de símbolos disponíveis na Hyperliquid
symbols_hype = [item['symbol'] for item in exchange.fetchSwapMarkets()]

# Conecta ao banco e busca os últimos timestamps
with engine.connect() as conn:
    latest_dates = pd.read_sql("""
        SELECT symbol, MAX(date) as last_date
        FROM ohlcv
        GROUP BY symbol
    """, conn)

# Converte datas para datetime e calcula o 'since' em milissegundos
latest_dates['last_date'] = pd.to_datetime(latest_dates['last_date'])
latest_dates['since'] = (latest_dates['last_date'] + pd.Timedelta(minutes=15)).astype(np.int64) // 10**6

# Cria o mapeamento symbol → since, ajustando o sufixo
symbol_since_map = dict(zip(latest_dates['symbol'], latest_dates['since']))

# Lista para armazenar os dados
all_data = []

# Loop apenas nos símbolos que já estão no banco
symbols_in_db = [s for s in symbols_hype if s.replace('/USDC:USDC', '') in symbol_since_map]

for symbol in symbols_in_db:
    try:
        # Remove sufixo para buscar 'since' correto
        databank_symbol = symbol.replace('/USDC:USDC', '')
        since = symbol_since_map.get(databank_symbol)

        if since is None:
            continue  # Pula se não tiver timestamp de referência

        # Busca os dados
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', since=since)

        for entry in ohlcv:
            all_data.append({
                'date': pd.to_datetime(entry[0], unit='ms'),
                'symbol': symbol,
                'open': entry[1],
                'high': entry[2],
                'low': entry[3],
                'close': entry[4],
                'volume': entry[5],
            })

    except Exception as e:
        logging.error(f"Erro ao buscar dados de {symbol}: {e}", exc_info=True)

# Constrói o DataFrame final
df = pd.DataFrame(all_data)
df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]


################################### SANITY CHECK - PREÇOS ###################################################

daily_returns = np.log(price.resample('D')).diff()

daily_std = daily_returns.ewm(span=35).std()

limit_change = daily_std.iloc[-1] * 16

last_price = price.iloc[-1]


# Função para aplicar o sanity check a uma coluna
def sanitize_column(col):
    # Cria uma matriz pivotada para comparação vetorizada
    pivot = df.pivot(columns='symbol', values=col)

    # Calcula retornos relativos diários
    rel_change = pivot.pct_change()

    # Substitui valores além do limite (para cima ou para baixo)
    mask = rel_change.abs() > (limit_change / last_price)

    # Substitui por último preço diário
    pivot[mask] = last_price

    # Reconstrói o DataFrame original
    return pivot.stack().rename(col)

# Aplica o sanity check para cada uma das colunas de preço
sanitized_cols = []
for col in ['open', 'high', 'low', 'close']:
    sanitized_col = sanitize_column(col)
    sanitized_cols.append(sanitized_col)

# Junta os resultados sanitizados
sanitized_df = pd.concat(sanitized_cols, axis=1)

# Restaura o índice original
df.update(sanitized_df)
df.reset_index(inplace=True)
df['symbol'] = df['symbol'].str.replace('/USDC:USDC', '', regex=True).str.strip()


###################################### INSERIR OS DADOS DE PREÇOS NO BANCO ####################################

df.to_sql('ohlcv', con=engine, index=False, if_exists='append')

# ajustar price para incluir dados recentes, sem precisar fazer uma nova consulta no banco
price = pd.concat([price, sass_crypto.format_from_database(df[['date', 'symbol', 'close']])])
price.index = pd.to_datetime(price.index)
price = price[~price.index.duplicated(keep='last')]
price = price.sort_index()


symbols = price.columns
##################### FILTRAR O UNIVERSO DE ATIVOS NEGOCIÁVEIS ##################################


# Número mínimo de observações exigidas
min_obs = 365 * 96

# Conta os valores não-nulos por coluna (ativo)
valid_counts = price.notna().sum()

# Filtra os ativos com mais de min_obs observações válidas
valid_symbols = valid_counts[valid_counts > min_obs].index.tolist()


volume = volume[valid_symbols]


# Última linha: os 20 menores ativos
n_symbols = 10

trading_universe = volume[-(90*96):].mean().nlargest(n_symbols).index.tolist()

filtered_volume = volume[trading_universe]

filtered_price = price[trading_universe]


########################################## CALCULAR RISK WEIGHTS ######################################

returns_15min = np.log(filtered_price.ffill()).diff()

# Inverse Volatility weights
vola_15min = returns_15min.ewm(span=35*96).std().iloc[-1]

inverse_vola = 1/vola_15min

inverse_vola_weights = inverse_vola/inverse_vola.sum()

# ADVT weights
adtv =  filtered_volume[-(63*96):].mean()

adtv_weights = adtv/adtv.sum()

# Risk Weights
risk_weights = ((adtv_weights + inverse_vola_weights)/2)




#################################### CALCULAR VOLATILITY REGIME MULTIPLIER ######################################

vol_regime_multiplier = sass_crypto.calculate_vrm(returns_15min)


#################################### CALCULAR INSTRUMENT DIVERSIFICATION MULTIPLIER ######################################

idm = sass_crypto.calculate_idm(returns_15min, risk_weights)


#################################### CALCULAR FORECASTS ######################################

# EWMAC
ewmac_2_8 = sass_crypto.ewmac(filtered_price, Lfast=2, vol_regime_multiplier=vol_regime_multiplier)

ewmac_4_16 = sass_crypto.ewmac(filtered_price, Lfast=4, vol_regime_multiplier=vol_regime_multiplier)

ewmac_8_32 = sass_crypto.ewmac(filtered_price, Lfast=8, vol_regime_multiplier=vol_regime_multiplier)

ewmac_16_64 = sass_crypto.ewmac(filtered_price, Lfast=16, vol_regime_multiplier=vol_regime_multiplier)

# Breakout
breakout_5 = sass_crypto.breakout(filtered_price, 5, vol_regime_multiplier)

breakout_10 = sass_crypto.breakout(filtered_price, 10, vol_regime_multiplier)

breakout_20 = sass_crypto.breakout(filtered_price, 20, vol_regime_multiplier)

breakout_40 = sass_crypto.breakout(filtered_price, 40, vol_regime_multiplier)

# Combined Forecast
forecast_scalar = 1.24
combined_forecast = (((ewmac_2_8 * 0.125) + 
               (ewmac_4_16 * 0.125) + 
               (ewmac_8_32 * 0.125) + 
               (ewmac_16_64 * 0.125) + 
                 (breakout_5 * 0.125) +
                  (breakout_10 * 0.125) +
                  (breakout_20 * 0.125) +
                  (breakout_40 * 0.125)) * forecast_scalar).clip(-20, 20)


################################ CALCULAR POSIÇÕES ÓTIMAS #####################################

account_info = exchange.fetch_balance()

capital = account_info.get('USDC', {}).get('total', None)
if capital is None:
    time.sleep(20)
    account_info = exchange.fetch_balance()  # essa linha estava faltando
    capital = account_info.get('USDC', {}).get('total', None)
    if capital is None:
        raise ValueError("Erro ao obter o capital em USDC.")



vol_target = 0.5

percent_vol = returns_15min.ewm(span=20*96).std() * np.sqrt(365*96)

last_percent_vol= percent_vol.iloc[-1]

last_price = filtered_price.iloc[-1]

# Posições ideais
n_contracts = (capital * combined_forecast * idm * risk_weights * vol_target) / (10 * last_percent_vol * last_price)

buffer_width = (0.1 * capital * idm * risk_weights * vol_target) / (percent_vol * price)

upper_buffer = round(n_contracts + buffer_width)

lower_buffer = round(n_contracts - buffer_width)


# Posições atuais
actual_positions_dict = exchange.fetchPositions()

def extract_position_size(p):
    try:
        return float(p['info']['position']['szi']) * (1 if p['side'] == 'long' else -1)
    except Exception as e:
        logging.warning(f"Erro ao interpretar posição de {p.get('symbol', '?')}: {e}")
        return 0.0

actual_positions = pd.Series({
    p['symbol']: extract_position_size(p)
    for p in actual_positions_dict
})


# Posições ótimas
optimal_positions = n_contracts.where(
    (actual_positions < lower_buffer) | (actual_positions > upper_buffer),
    actual_positions
)


####################################### RISK OVERLAY ####################################################

# Leverage
leverage = (optimal_positions.abs() * last_price).sum()

max_risk_leverage = 2

leverage_risk_multiplier = min(1, max_risk_leverage/leverage)

if leverage == 0:
    leverage_risk_multiplier = 1
else:
    leverage_risk_multiplier = min(1, max_risk_leverage / leverage)



# Expected Risk
dollar_weights = (optimal_positions.abs() * last_price)

dollar_weights = dollar_weights.reindex(percent_vol.index)

cmatrix = returns_15min.iloc[-20*96:].corr().values

sigma = np.diag(percent_vol.values).dot(cmatrix).dot(np.diag(percent_vol.values))

portfolio_variance = dollar_weights.dot(sigma).dot(dollar_weights.transpose())

portfolio_std = portfolio_variance ** .5

expected_risk_multiplier = min(1, 1.25/portfolio_std)

# Risk Multiplier
risk_overlay_multiplier = min(leverage_risk_multiplier, expected_risk_multiplier)
optimal_positions = optimal_positions * risk_overlay_multiplier

logging.info(f"Risk overlay aplicado: leverage_multiplier={leverage_risk_multiplier:.4f}, expected_risk_multiplier={expected_risk_multiplier:.4f}, total={risk_overlay_multiplier:.4f}")


# Limit Exposure
limit_position = ((optimal_positions * last_price) / capital).clip(-0.2, 0.4)

optimal_positions = (limit_position * capital) / last_price


################################ EXECUÇÃO DE ORDENS #####################################

# 1. Identifica ativos a zerar (presentes apenas em actual_positions)

optimal_positions.index = optimal_positions.index.map(lambda x: f"{x}/USDC:USDC")


symbols_to_close = actual_positions.index.difference(optimal_positions.index)

# 2. Gera ordens de zeragem (reduceOnly = True)
orders = []
for symbol in symbols_to_close:
    amount = actual_positions[symbol]
    if amount == 0:
        continue

    orders.append({
        "symbol": symbol,
        "type": "market",
        "side": "sell" if amount > 0 else "buy",
        "amount": abs(amount),
        "params": {
            "reduceOnly": True
        }
    })

# 3. Remove esses ativos antes de calcular os trades restantes
actual_positions = actual_positions.drop(symbols_to_close)

# 4. Garante alinhamento dos índices
all_symbols = optimal_positions.index.union(actual_positions.index)
optimal_positions = optimal_positions.reindex(all_symbols, fill_value=0)
actual_positions = actual_positions.reindex(all_symbols, fill_value=0)

# 5. Calcula os trades
trades = optimal_positions - actual_positions

# 6. Calcula o valor financeiro por trade
fin_amount = (trades * price).abs()

# 7. Ajusta para respeitar o mínimo financeiro (ordens < $5 são ignoradas, $5–$10 são arredondadas)
adjusted_trades = trades.where(~((fin_amount > 0) & (fin_amount < 5)), 0)

# Substitui onde o valor financeiro está entre 5 e 10 (exclusive) por uma ordem mínima viável
min_amount = (10 / price).where(trades != 0, 0) * np.sign(trades)
adjusted_trades = adjusted_trades.where(~((fin_amount >= 5) & (fin_amount < 10)), min_amount)

# 8. Monta ordens normais (market)
for symbol, amount in adjusted_trades.items():
    if amount == 0 or pd.isna(amount):
        continue

    orders.append({
        "symbol": symbol,
        "type": "market",
        "side": "buy" if amount > 0 else "sell",
        "amount": abs(amount),
    })


execute_orders = exchange.createOrders(orders)


time.sleep(20)


# Função para verificar se a ordem foi totalmente executada
def was_filled(order):
    try:
        filled_sz = float(order['info']['filled']['totalSz'])
        requested_sz = float(order['amount'])
        return abs(requested_sz - filled_sz) < 1e-6
    except (KeyError, TypeError, ValueError):
        return False

# Filtra ordens parcialmente executadas
partial_fills = [o for o in execute_orders if not was_filled(o)]

# Reporta
if partial_fills:
    logging.warning("Ordens não totalmente executadas:")
    for o in partial_fills:
        symbol = o.get('symbol', 'UNKNOWN')
        amount = o.get('amount', '?')
        filled = o.get('info', {}).get('filled', {}).get('totalSz', '?')
        logging.warning(f"{symbol}: solicitado={amount} | executado={filled}")

else:
    logging.info(f"Ordens executadas com sucesso: {len(execute_orders)}")


updated_positions = exchange.fetchPositions()


execute_orders_df = pd.DataFrame(execute_orders)
execute_orders_df.to_sql('orders', con=engine, index=False, if_exists='append')

updated_positions_df = pd.DataFrame(updated_positions)
updated_positions_df.to_sql('positions', con=engine, index=False, if_exists='append')

logging.info("Execução completa com sucesso.")
logging.info(f"Tempo total de execução: {round(time.time() - start_time, 2)} segundos.")
