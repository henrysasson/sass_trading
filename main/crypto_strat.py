from dotenv import load_dotenv
load_dotenv()
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import os
import pysass_crypto as sass_crypto
import logging
import gc
import psutil

start_time = time.time()

# Monitoramento de memória
def log_memory_usage(step):
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    logging.info(f"Memória após {step}: {memory_mb:.1f}MB")

############################ CONFIGURAÇÕES #################################################################################
exchange = ccxt.hyperliquid({
    "walletAddress": "0x973A318a9984bA6B3C6965cBF86f27546FA91C88",
    "privateKey": "0x615a7c174a1ad31ddb606b8a7229df1b8e5db786f5aba2bb52b7d9688f84d58e",
})

# Configuração do logger
log_filename = f"trading_log_{datetime.today().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASS']
DB_HOST = os.environ['DB_HOST']
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ['DB_NAME']

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

###########################################################################################################
# CARREGAMENTO INICIAL: Símbolos já estão limpos no banco
initial_date = (datetime.today() - timedelta(days=370)).strftime('%Y-%m-%d')

# Carregar dados do banco (símbolos já em formato limpo)
query = f"""
    SELECT date, symbol, close, volume 
    FROM ohlcv 
    WHERE date >= '{initial_date}'
    ORDER BY date, symbol
"""

dtype_dict = {
    'symbol': 'category',
    'close': 'float32',
    'volume': 'float32'
}

data = pd.read_sql_query(query, engine, dtype=dtype_dict)
# Conversão robusta de datetime para lidar com formatos mistos
data['date'] = pd.to_datetime(data['date'], format='mixed', errors='coerce')
# Remover linhas com datas inválidas
data = data.dropna(subset=['date'])
log_memory_usage("carregamento inicial")

# Usar dados direto para cálculos (símbolos já limpos)
price = data.pivot_table(index='date', columns='symbol', values='close', aggfunc='last', observed=True).astype('float32')
volume_raw = data.pivot_table(index='date', columns='symbol', values='volume', aggfunc='last', observed=True).astype('float32')

# Limpar dados originais imediatamente
del data
gc.collect()
log_memory_usage("após pivot")

# Volume em dólares
volume = volume_raw * price
del volume_raw
gc.collect()

###################################### PUXAR OS PREÇOS DA EXCHANGE ####################################
symbols = price.columns
symbols_hype = [item['symbol'] for item in exchange.fetchSwapMarkets()]

# OTIMIZAÇÃO 3: Processar exchange data em chunks menores
symbols_hype = [item['symbol'] for item in exchange.fetchSwapMarkets()]

for symbol in symbols_hype:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m')
        
        if not isinstance(ohlcv, list) or len(ohlcv) == 0 or not all(isinstance(row, list) and len(row) == 6 for row in ohlcv):
            raise ValueError(f"Formato inesperado de OHLCV para {symbol}: {ohlcv}")
    
        df_chunk = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_chunk['date'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
        clean_symbol = symbol.replace('/USDC:USDC', '').replace('/USD:USD', '').replace('/USDT:USDT', '').strip()
        df_chunk['symbol'] = clean_symbol
        df_chunk = df_chunk[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

        df_chunk.to_sql('ohlcv', con=engine, index=False, if_exists='append')

        del df_chunk
        gc.collect()

    except Exception as e:
        logging.error(f"Erro ao buscar dados de {symbol}: {e}", exc_info=True)

log_memory_usage("após fetch exchange")

# OTIMIZAÇÃO 4: Sanity check mais eficiente
data_recent = (datetime.utcnow() - timedelta(days=4)).strftime('%Y-%m-%d %H:%M:%S')
query_recent = f"""
    SELECT date, symbol, open, high, low, close, volume FROM ohlcv 
    WHERE date >= '{data_recent}'
    ORDER BY date, symbol
"""

df = pd.read_sql_query(query_recent, engine, dtype={
    'symbol': 'category', 'open': 'float32', 'high': 'float32',
    'low': 'float32', 'close': 'float32', 'volume': 'float32'
})
# Conversão robusta de datetime
df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
df = df.dropna(subset=['date'])

df.to_sql('ohlcv', con=engine, index=False, if_exists='append')

# OTIMIZAÇÃO 5: Atualizar price de forma mais eficiente
price_update = df.pivot_table(
    index='date', columns='symbol', values='close', aggfunc='last'
).astype('float32')

price = price_update.combine_first(price).sort_index()
price = price.loc[~price.index.duplicated(keep='last')]

# Atualizar volume
volume_update = df.pivot_table(
    index='date', columns='symbol', values='volume', aggfunc='last'
).astype('float32')
volume_update = volume_update * price_update

volume = volume_update.combine_first(volume).sort_index()
volume = volume.loc[~volume.index.duplicated(keep='last')]

# Limpar dados temporários
del df, price_update, volume_update
gc.collect()
log_memory_usage("após sanity check")

##################### FILTRAR O UNIVERSO DE ATIVOS NEGOCIÁVEIS (OTIMIZADO) ##################################
min_obs = 365 * 96
valid_counts = price.count()  # count() é mais eficiente que notna().sum()
valid_symbols = valid_counts[valid_counts > min_obs].index.tolist()

# Filtrar dados
volume = volume[valid_symbols]
price = price[valid_symbols]

# OTIMIZAÇÃO 6: Seleção de universo mais eficiente
n_symbols = 10
recent_volume = volume.tail(90*96)  # Usar tail() é mais eficiente
trading_universe = recent_volume.mean().nlargest(n_symbols).index.tolist()

filtered_volume = volume[trading_universe].copy()
filtered_price = price[trading_universe].copy()

# Limpar dados grandes não utilizados
del volume, price, recent_volume
gc.collect()
log_memory_usage("após filtros")

########################################## CALCULAR RISK WEIGHTS (OTIMIZADO) ######################################
returns_15min = np.log(filtered_price.ffill()).diff().astype('float32')

# Inverse Volatility weights
vola_15min = returns_15min.ewm(span=35*96).std().iloc[-1]
inverse_vola = 1/vola_15min
inverse_vola_weights = inverse_vola/inverse_vola.sum()

# ADVT weights
adtv = filtered_volume.tail(63*96).mean()  # Usar tail() é mais eficiente
adtv_weights = adtv/adtv.sum()

# Risk Weights
risk_weights = ((adtv_weights + inverse_vola_weights)/2)

log_memory_usage("após risk weights")

#################################### CALCULAR MULTIPLICADORES ######################################
vol_regime_multiplier = sass_crypto.calculate_vrm(returns_15min)
idm = sass_crypto.calculate_idm(returns_15min, risk_weights)

#################################### CALCULAR FORECASTS (OTIMIZADO) ######################################
# Calcular todos os forecasts de uma vez para evitar recálculos
forecasts = {}
ewmac_params = [(2, 0.125), (4, 0.125), (8, 0.125), (16, 0.125)]
breakout_params = [(5, 0.125), (10, 0.125), (20, 0.125), (40, 0.125)]

for Lfast, weight in ewmac_params:
    forecasts[f'ewmac_{Lfast}'] = sass_crypto.ewmac(filtered_price, Lfast, vol_regime_multiplier=vol_regime_multiplier) * weight

for horizon, weight in breakout_params:
    forecasts[f'breakout_{horizon}'] = sass_crypto.breakout(filtered_price, horizon, vol_regime_multiplier) * weight

# Combined Forecast
forecast_scalar = 1.24
combined_forecast = sum(forecasts.values()) * forecast_scalar
combined_forecast = combined_forecast.clip(-20, 20)

del forecasts
gc.collect()
log_memory_usage("após forecasts")

################################ CALCULAR POSIÇÕES ÓTIMAS (OTIMIZADO) #####################################
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

buffer_width = (0.1 * capital * idm * risk_weights * vol_target) / (last_percent_vol * last_price)

upper_buffer = round(n_contracts + buffer_width)

lower_buffer = round(n_contracts - buffer_width)


# Posições atuais
actual_positions_dict = exchange.fetchPositions()

actual_positions = pd.Series({
    p['symbol']: float(p['info']['position']['szi']) * (1 if p['side'] == 'long' else -1)
    for p in actual_positions_dict
})


# Posições ótimas
optimal_positions = n_contracts.where(
    (actual_positions < lower_buffer) | (actual_positions > upper_buffer),
    actual_positions
)

####################################### RISK OVERLAY (OTIMIZADO) ####################################################
# Leverage
leverage = (optimal_positions.abs() * last_price).sum()
max_risk_leverage = 2

if leverage == 0:
    leverage_risk_multiplier = 1
else:
    leverage_risk_multiplier = min(1, max_risk_leverage / leverage)

# Expected Risk (versão otimizada)
dollar_weights = (optimal_positions.abs() * last_price)
dollar_weights = dollar_weights.reindex(last_percent_vol.index, fill_value=0)

# Usar apenas dados recentes para correlação (reduz uso de memória)
recent_returns = returns_15min.tail(20*96)
cmatrix = recent_returns.corr().values

sigma = np.outer(last_percent_vol.values, last_percent_vol.values) * cmatrix
portfolio_variance = dollar_weights.values.dot(sigma).dot(dollar_weights.values)
portfolio_std = np.sqrt(portfolio_variance)

expected_risk_multiplier = min(1, 1.25/portfolio_std)

# Risk Multiplier
risk_overlay_multiplier = min(leverage_risk_multiplier, expected_risk_multiplier)
optimal_positions = optimal_positions * risk_overlay_multiplier

logging.info(f"Risk overlay aplicado: leverage_multiplier={leverage_risk_multiplier:.4f}, expected_risk_multiplier={expected_risk_multiplier:.4f}, total={risk_overlay_multiplier:.4f}")

# Limit Exposure
limit_position = ((optimal_positions * last_price) / capital).clip(-0.2, 0.4)
optimal_positions = (limit_position * capital) / last_price

# Log informativo das posições ótimas
logging.info(f"Posições ótimas:\n{optimal_positions.to_string()}")


################################ EXECUÇÃO DE ORDENS (OTIMIZADO) #####################################

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
execute_orders_df.to_sql('orders', con=database, index=False, if_exists='append')

updated_positions_df = pd.DataFrame(updated_positions)
updated_positions_df.to_sql('positions', con=database, index=False, if_exists='append')

logging.info("Execução completa com sucesso.")
logging.info(f"Tempo total de execução: {round(time.time() - start_time, 2)} segundos.")
