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
        if not ohlcv:
            continue

        df_chunk = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_chunk['date'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
        clean_symbol = symbol.replace('/USDC:USDC', '').replace('/USD:USD', '').replace('/USDT:USDT', '').strip()
        df_chunk['symbol'] = clean_symbol
        df_chunk = df_chunk[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

        df_chunk.to_sql('ohlcv', con=database, index=False, if_exists='append')

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
    account_info = exchange.fetch_balance()
    capital = account_info.get('USDC', {}).get('total', None)
    if capital is None:
        raise ValueError("Erro ao obter o capital em USDC.")

vol_target = 0.5

# OTIMIZAÇÃO 7: Cálculos vetorizados para posições
percent_vol = returns_15min.ewm(span=20*96).std() * np.sqrt(365*96)
last_percent_vol = percent_vol.iloc[-1]
last_price = filtered_price.iloc[-1]

# Posições ideais (cálculo vetorizado)
n_contracts = (capital * combined_forecast * idm * risk_weights * vol_target) / (10 * last_percent_vol * last_price)
buffer_width = (0.1 * capital * idm * risk_weights * vol_target) / (last_percent_vol * last_price)

upper_buffer = n_contracts + buffer_width
lower_buffer = n_contracts - buffer_width

# Posições atuais
actual_positions_dict = exchange.fetchPositions()
actual_positions_raw = pd.Series({
    p['symbol']: float(p['info']['position']['szi']) * (1 if p['side'] == 'long' else -1)
    for p in actual_positions_dict if float(p['info']['position']['szi']) != 0  # Apenas posições não-zero
}, dtype='float32')

# CORREÇÃO ROBUSTA: Mapear símbolos da exchange para formato do banco
def map_exchange_to_clean_symbol(exchange_symbol):
    """Converte símbolo da exchange para formato limpo do banco"""
    return exchange_symbol.replace('/USDC:USDC', '').replace('/USD:USD', '').replace('/USDT:USDT', '').strip()

def map_clean_to_exchange_symbol(clean_symbol):
    """Converte símbolo limpo para formato da exchange (padrão USDC)"""
    return f"{clean_symbol}/USDC:USDC"

# Converter posições atuais para formato limpo
actual_positions_clean = pd.Series(dtype='float32')
for symbol_exchange, position in actual_positions_raw.items():
    symbol_clean = map_exchange_to_clean_symbol(symbol_exchange)
    actual_positions_clean[symbol_clean] = position

# Debug: verificar alinhamento de símbolos
n_contracts_symbols = set(n_contracts.index)
actual_positions_symbols = set(actual_positions_clean.index)
common_symbols = n_contracts_symbols.intersection(actual_positions_symbols)
only_in_calc = n_contracts_symbols - actual_positions_symbols
only_in_positions = actual_positions_symbols - n_contracts_symbols

logging.info(f"Símbolos em comum: {len(common_symbols)}")
logging.info(f"Apenas em cálculos: {only_in_calc}")
logging.info(f"Apenas em posições: {only_in_positions}")

# Trabalhar apenas com símbolos que estão no universo de trading ou que têm posições
relevant_symbols = n_contracts_symbols.union(actual_positions_symbols)

# Alinhar todos os dados para símbolos relevantes
n_contracts_aligned = n_contracts.reindex(relevant_symbols, fill_value=0)
actual_positions_aligned = actual_positions_clean.reindex(relevant_symbols, fill_value=0)
upper_buffer_aligned = upper_buffer.reindex(relevant_symbols, fill_value=0)
lower_buffer_aligned = lower_buffer.reindex(relevant_symbols, fill_value=0)

# Posições ótimas (apenas rebalancear onde necessário)
optimal_positions = n_contracts_aligned.where(
    (actual_positions_aligned < lower_buffer_aligned) | (actual_positions_aligned > upper_buffer_aligned),
    actual_positions_aligned
)

# Zerar posições de símbolos que não fazem mais parte do universo de trading
for symbol in only_in_positions:
    if symbol not in n_contracts_symbols:
        optimal_positions[symbol] = 0  # Forçar fechamento

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

################################ EXECUÇÃO DE ORDENS (OTIMIZADO) #####################################

# Criar mapeamento reverso: clean_symbol -> exchange_symbol baseado nas posições atuais
clean_to_exchange_map = {}
for exchange_symbol in actual_positions_raw.index:
    clean_symbol = map_exchange_to_clean_symbol(exchange_symbol)
    clean_to_exchange_map[clean_symbol] = exchange_symbol

# Para símbolos novos (não em posições atuais), usar o formato padrão
for clean_symbol in optimal_positions.index:
    if clean_symbol not in clean_to_exchange_map:
        clean_to_exchange_map[clean_symbol] = map_clean_to_exchange_symbol(clean_symbol)

# Converter optimal_positions para formato da exchange
optimal_positions_exchange = pd.Series(dtype='float32')
for clean_symbol, position in optimal_positions.items():
    exchange_symbol = clean_to_exchange_map.get(clean_symbol, map_clean_to_exchange_symbol(clean_symbol))
    optimal_positions_exchange[exchange_symbol] = position

# Identificar símbolos para fechar (estão em posições atuais mas não em optimal)
symbols_to_close = set(actual_positions_raw.index) - set(optimal_positions_exchange.index)

# Gerar ordens de fechamento
orders = []
for symbol in symbols_to_close:
    amount = actual_positions_raw[symbol]
    if abs(amount) < 1e-6:  # Ignorar posições muito pequenas
        continue
    
    orders.append({
        "symbol": symbol,
        "type": "market",
        "side": "sell" if amount > 0 else "buy",
        "amount": abs(amount),
        "params": {"reduceOnly": True}
    })

logging.info(f"Ordens de fechamento: {len(orders)} para símbolos {list(symbols_to_close)}")

# Calcular trades para símbolos relevantes
trades_exchange = pd.Series(dtype='float32')
for exchange_symbol, optimal_pos in optimal_positions_exchange.items():
    current_pos = actual_positions_raw.get(exchange_symbol, 0)
    trade_size = optimal_pos - current_pos
    
    if abs(trade_size) > 1e-6:  # Apenas trades significativos
        trades_exchange[exchange_symbol] = trade_size

# Filtrar trades por valor mínimo
trades_financial_value = pd.Series(dtype='float32')
for exchange_symbol, trade_size in trades_exchange.items():
    clean_symbol = map_exchange_to_clean_symbol(exchange_symbol)
    price = last_price.get(clean_symbol, 1)  # Default price se não encontrar
    fin_value = abs(trade_size * price)
    trades_financial_value[exchange_symbol] = fin_value

# Ajustar trades baseado no valor financeiro
adjusted_trades = pd.Series(dtype='float32')
for exchange_symbol, trade_size in trades_exchange.items():
    fin_value = trades_financial_value[exchange_symbol]
    clean_symbol = map_exchange_to_clean_symbol(exchange_symbol)
    price = last_price.get(clean_symbol, 1)
    
    if fin_value < 5:
        # Ignorar trades muito pequenos
        continue
    elif fin_value < 10:
        # Ajustar para valor mínimo
        min_size = 11 / price
        adjusted_trades[exchange_symbol] = min_size * (1 if trade_size > 0 else -1)
    else:
        # Manter trade original
        adjusted_trades[exchange_symbol] = trade_size

# Gerar ordens normais
for exchange_symbol, amount in adjusted_trades.items():
    if abs(amount) < 1e-6:  # Ignorar valores muito pequenos
        continue
    
    orders.append({
        "symbol": exchange_symbol,  # Já está no formato correto da exchange
        "type": "market",
        "side": "buy" if amount > 0 else "sell",
        "amount": abs(amount),
    })

logging.info(f"Total de ordens: {len(orders)} (fechamento + novas posições)")

# Executar ordens
execute_orders = exchange.createOrders(orders) if orders else []

time.sleep(20)

# Verificação de execução
def was_filled(order):
    try:
        filled_sz = float(order['info']['filled']['totalSz'])
        requested_sz = float(order['amount'])
        return abs(requested_sz - filled_sz) < 1e-6
    except (KeyError, TypeError, ValueError):
        return False

# Relatório de execução
if execute_orders:
    partial_fills = [o for o in execute_orders if not was_filled(o)]
    
    if partial_fills:
        logging.warning("Ordens não totalmente executadas:")
        for o in partial_fills:
            symbol = o.get('symbol', 'UNKNOWN')
            amount = o.get('amount', '?')
            filled = o.get('info', {}).get('filled', {}).get('totalSz', '?')
            logging.warning(f"{symbol}: solicitado={amount} | executado={filled}")
    else:
        logging.info(f"Ordens executadas com sucesso: {len(execute_orders)}")

# Salvar resultados
updated_positions = exchange.fetchPositions()

if execute_orders:
    execute_orders_df = pd.DataFrame(execute_orders)
    execute_orders_df.to_sql('orders', con=engine, index=False, if_exists='append')

if updated_positions:
    updated_positions_df = pd.DataFrame(updated_positions)
    updated_positions_df.to_sql('positions', con=engine, index=False, if_exists='append')

# Limpeza final
gc.collect()
log_memory_usage("final")

logging.info("Execução completa com sucesso.")
logging.info(f"Tempo total de execução: {round(time.time() - start_time, 2)} segundos.")
