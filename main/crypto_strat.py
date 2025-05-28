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
import psutil  # Para monitorar memória

start_time = time.time()

############################ CONFIGURAÇÕES OTIMIZADAS #################################################################################

# Configurar pandas para usar menos memória
pd.set_option('mode.copy_on_write', True)
pd.set_option('compute.use_numba', True)

exchange = ccxt.hyperliquid({
    "walletAddress": "0x973A318a9984bA6B3C6965cBF86f27546FA91C88",
    "privateKey": "0x615a7c174a1ad31ddb606b8a7229df1b8e5db786f5aba2bb52b7d9688f84d58e",
})

# Logging otimizado
log_filename = f"trading_log_{datetime.today().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Variáveis de ambiente
DB_USER = os.environ['DB_USER']
DB_PASS = os.environ['DB_PASS']
DB_HOST = os.environ['DB_HOST']
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ['DB_NAME']

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)

def log_memory_usage(step_name):
    """Função para monitorar uso de memória"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"{step_name}: Uso de memória = {memory_mb:.1f} MB")
    return memory_mb

###########################################################################################################
# OTIMIZAÇÃO 1: Query mais específica e chunked processing

log_memory_usage("Início")

# Reduzir janela de dados históricos (300 dias ao invés de 370)
initial_date = (datetime.today() - timedelta(days=300)).strftime('%Y-%m-%d')

# Query otimizada com LIMIT para controlar memória
query = f"""
    SELECT date, symbol, close, volume 
    FROM ohlcv 
    WHERE date >= '{initial_date}'
    ORDER BY date DESC
    LIMIT 2000000
"""

# Leitura chunked para economizar memória
chunk_size = 100000
chunks = []
for chunk in pd.read_sql_query(query, engine, chunksize=chunk_size):
    chunk = sass_crypto.optimize_dataframe_memory(chunk)
    chunks.append(chunk)

data = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

log_memory_usage("Após carregar dados")

###########################################################################################################
# OTIMIZAÇÃO 2: Processamento mais eficiente de volume e preço

# Processar volume e preço com tipos otimizados
volume = sass_crypto.format_from_database(data[['date', 'symbol', 'volume']].copy())
price = sass_crypto.format_from_database(data[['date', 'symbol', 'close']].copy())

# Liberar memória do DataFrame original
del data
gc.collect()

# Volume ajustado
volume = volume.multiply(price, fill_value=0).astype('float32')

log_memory_usage("Após processar volume/preço")

###################################### OTIMIZAÇÃO 3: Fetch de dados da exchange em batches ####################################

symbols = price.columns.tolist()
symbols_hype = [item['symbol'] for item in exchange.fetchSwapMarkets()]

# Processar em batches menores para controlar memória
batch_size = 20
new_data_chunks = []

for i in range(0, len(symbols_hype), batch_size):
    batch_symbols = symbols_hype[i:i+batch_size]
    
    for symbol in batch_symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)  # Limit para controlar
            if not ohlcv:
                continue

            df_chunk = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_chunk['date'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
            df_chunk['symbol'] = symbol
            df_chunk = df_chunk[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            # Otimizar tipos
            df_chunk = sass_crypto.optimize_dataframe_memory(df_chunk)
            new_data_chunks.append(df_chunk)

        except Exception as e:
            logging.error(f"Erro ao buscar dados de {symbol}: {e}")
    
    # Força garbage collection a cada batch
    gc.collect()

# Consolidar novos dados
if new_data_chunks:
    new_data = pd.concat(new_data_chunks, ignore_index=True)
    del new_data_chunks
    gc.collect()
else:
    new_data = pd.DataFrame()

log_memory_usage("Após fetch da exchange")

################################### OTIMIZAÇÃO 4: Sanity check mais eficiente ###################################################

if not new_data.empty:
    # Sanity check otimizado
    price_sorted = price.sort_index()
    price_sorted = price_sorted[~price_sorted.index.duplicated(keep='last')]
    
    # Usar apenas últimos 30 dias para calcular volatilidade (ao invés de todos os dados)
    recent_price = price_sorted.tail(30*96)
    daily_close = recent_price.resample('1D').last()
    daily_returns = np.log(daily_close).diff()
    daily_std = daily_returns.ewm(span=20).std()  # Janela menor
    limit_change = daily_std.iloc[-1] * 10  # Limite menos restritivo
    
    # Aplicar sanity check apenas nos novos dados
    def sanitize_new_data(df, limit_change):
        df_pivot = df.pivot_table(index='date', columns='symbol', values=['open', 'high', 'low', 'close'], aggfunc='last')
        
        for col in ['open', 'high', 'low', 'close']:
            price_col = df_pivot[col]
            rel_change = price_col.pct_change()
            
            # Aplicar limite mais flexível
            mask = rel_change.abs() > 0.5  # 50% change limit
            price_col[mask] = price_col.shift(1)[mask]
        
        return df_pivot.stack().reset_index()
    
    if len(new_data) > 0:
        sanitized_new_data = sanitize_new_data(new_data, limit_change)
        # Inserir no banco
        sanitized_new_data.to_sql('ohlcv', con=engine, index=False, if_exists='append', method='multi')
        
        # Atualizar price com novos dados
        price_update = sass_crypto.format_from_database(sanitized_new_data[['date', 'symbol', 'close']])
        price = price_update.combine_first(price).sort_index()
        price = price.loc[~price.index.duplicated(keep='last')]
        
        del sanitized_new_data, price_update
        gc.collect()

log_memory_usage("Após sanity check")

##################### OTIMIZAÇÃO 5: Filtrar universo de forma mais eficiente ##################################

# Reduzir requisitos mínimos para incluir mais ativos
min_obs = 200 * 96  # Reduzido de 365*96

# Contar observações válidas de forma mais eficiente
valid_counts = (~price.isna()).sum()
valid_symbols = valid_counts[valid_counts > min_obs].index.tolist()

# Filtrar dados
volume_filtered = volume[valid_symbols].copy()
price_filtered = price[valid_symbols].copy()

# Reduzir universo de trading (menos ativos = menos cálculos)
n_symbols = 8  # Reduzido de 10
recent_volume = volume_filtered.tail(60*96)  # Últimos 60 dias ao invés de 90
trading_universe = recent_volume.mean().nlargest(n_symbols).index.tolist()

filtered_volume = volume_filtered[trading_universe].astype('float32')
filtered_price = price_filtered[trading_universe].astype('float32')

# Limpar dados não utilizados
del volume, price, volume_filtered, price_filtered, recent_volume
gc.collect()

log_memory_usage("Após filtrar universo")

########################################## OTIMIZAÇÃO 6: Risk weights mais eficiente ######################################

returns_15min = np.log(filtered_price.ffill()).diff().astype('float32')

# Usar janela menor para volatilidade
vola_15min = returns_15min.ewm(span=25*96).std().iloc[-1]  # Reduzido de 35*96
inverse_vola = 1/vola_15min
inverse_vola_weights = inverse_vola/inverse_vola.sum()

# ADTV com janela menor
adtv = filtered_volume.tail(45*96).mean()  # Reduzido de 63*96
adtv_weights = adtv/adtv.sum()

# Risk weights combinados
risk_weights = ((adtv_weights + inverse_vola_weights)/2).astype('float32')

log_memory_usage("Após calcular risk weights")

#################################### OTIMIZAÇÃO 7: Cálculos otimizados ######################################

# VRM otimizado
vol_regime_multiplier = sass_crypto.calculate_vrm(returns_15min)

# IDM otimizado
idm = sass_crypto.calculate_idm(returns_15min, risk_weights, lookback=200*96)  # Janela menor

log_memory_usage("Após VRM e IDM")

#################################### OTIMIZAÇÃO 8: Forecasts em paralelo ######################################

# Lista para armazenar forecasts
forecasts = []

# EWMAC forecasts
ewmac_params = [2, 4, 8, 16]
for lfast in ewmac_params:
    forecast = sass_crypto.ewmac(filtered_price, Lfast=lfast, vol_regime_multiplier=vol_regime_multiplier)
    forecasts.append(forecast * 0.125)  # Peso igual
    gc.collect()  # Limpar após cada cálculo

# Breakout forecasts
breakout_params = [5, 10, 20, 40]
for horizon in breakout_params:
    forecast = sass_crypto.breakout(filtered_price, horizon, vol_regime_multiplier)
    forecasts.append(forecast * 0.125)  # Peso igual
    gc.collect()

# Combined Forecast otimizado
forecast_scalar = 1.24
combined_forecast = sum(forecasts) * forecast_scalar
combined_forecast = combined_forecast.clip(-20, 20)

# Limpar forecasts individuais
del forecasts
gc.collect()

log_memory_usage("Após calcular forecasts")

################################ OTIMIZAÇÃO 9: Posições ótimas com menos cálculos #####################################

# Capital
account_info = exchange.fetch_balance()
capital = account_info.get('USDC', {}).get('total', None)
if capital is None:
    time.sleep(10)  # Reduzido de 20
    account_info = exchange.fetch_balance()
    capital = account_info.get('USDC', {}).get('total', None)
    if capital is None:
        raise ValueError("Erro ao obter o capital em USDC.")

vol_target = 0.4  # Reduzido de 0.5 para menos risco

# Volatilidade com janela menor
percent_vol = returns_15min.ewm(span=15*96).std() * np.sqrt(365*96)  # Reduzido de 20*96
last_percent_vol = percent_vol.iloc[-1]
last_price = filtered_price.iloc[-1]

# Posições ideais
n_contracts = (capital * combined_forecast * idm * risk_weights * vol_target) / (10 * last_percent_vol * last_price)

# Buffer simplificado
buffer_width = 0.05 * abs(n_contracts)  # Buffer de 5%
upper_buffer = n_contracts + buffer_width
lower_buffer = n_contracts - buffer_width

log_memory_usage("Após calcular posições")

####################################### OTIMIZAÇÃO 10: Risk overlay eficiente ####################################################

# Posições atuais
actual_positions_dict = exchange.fetchPositions()
actual_positions = pd.Series({
    p['symbol']: float(p['info']['position']['szi']) * (1 if p['side'] == 'long' else -1)
    for p in actual_positions_dict if float(p['info']['position']['szi']) != 0
}, dtype='float32')

# Posições ótimas com buffer
optimal_positions = n_contracts.where(
    (actual_positions.reindex(n_contracts.index, fill_value=0) < lower_buffer) | 
    (actual_positions.reindex(n_contracts.index, fill_value=0) > upper_buffer),
    actual_positions.reindex(n_contracts.index, fill_value=0)
)

# Risk overlay simplificado
leverage = (optimal_positions.abs() * last_price).sum() / capital
max_leverage = 1.5  # Reduzido de 2.0

leverage_multiplier = min(1, max_leverage/leverage) if leverage > 0 else 1

# Expected risk simplificado
portfolio_value = (optimal_positions.abs() * last_price).sum()
expected_risk_multiplier = min(1, capital * 1.0 / portfolio_value) if portfolio_value > 0 else 1

# Risk overlay final
risk_overlay_multiplier = min(leverage_multiplier, expected_risk_multiplier)
optimal_positions = optimal_positions * risk_overlay_multiplier

# Exposure limit
exposure_limit = 0.15  # Reduzido de 0.2
position_values = optimal_positions * last_price
exposure_ratios = position_values / capital
optimal_positions = optimal_positions.where(exposure_ratios.abs() <= exposure_limit, 
                                          np.sign(optimal_positions) * exposure_limit * capital / last_price)

log_memory_usage("Após risk overlay")

################################ OTIMIZAÇÃO 11: Execução otimizada #####################################

# Preparar símbolos para execução
optimal_positions.index = optimal_positions.index.map(lambda x: f"{x}/USDC:USDC")

# Identificar trades
all_symbols = optimal_positions.index.union(actual_positions.index)
optimal_aligned = optimal_positions.reindex(all_symbols, fill_value=0)
actual_aligned = actual_positions.reindex(all_symbols, fill_value=0)

trades = optimal_aligned - actual_aligned

# Filtrar trades significativos (valor mínimo $8)
min_trade_value = 8
price_aligned = last_price.reindex(trades.index.str.replace('/USDC:USDC', ''), fill_value=1)
price_aligned.index = trades.index

trade_values = (trades * price_aligned).abs()
significant_trades = trades[trade_values >= min_trade_value]

# Gerar ordens
orders = []
for symbol, amount in significant_trades.items():
    if abs(amount) < 1e-6:
        continue
    
    orders.append({
        "symbol": symbol,
        "type": "market",
        "side": "buy" if amount > 0 else "sell",
        "amount": abs(amount),
    })

# Executar ordens se houver
if orders:
    try:
        execute_orders = exchange.createOrders(orders)
        logging.info(f"Executadas {len(execute_orders)} ordens com sucesso")
        
        # Salvar no banco
        if execute_orders:
            execute_orders_df = pd.DataFrame(execute_orders)
            execute_orders_df.to_sql('orders', con=engine, index=False, if_exists='append')
            
    except Exception as e:
        logging.error(f"Erro na execução de ordens: {e}")
        execute_orders = []
else:
    logging.info("Nenhuma ordem necessária")
    execute_orders = []

log_memory_usage("Após execução")

# Salvar posições atualizadas
try:
    time.sleep(5)  # Reduzido
    updated_positions = exchange.fetchPositions()
    updated_positions_df = pd.DataFrame(updated_positions)
    updated_positions_df.to_sql('positions', con=engine, index=False, if_exists='append')
except Exception as e:
    logging.error(f"Erro ao salvar posições: {e}")

# Limpeza final de memória
gc.collect()

execution_time = time.time() - start_time
log_memory_usage("Final")
logging.info(f"Execução completa. Tempo total: {execution_time:.2f}s")
logging.info(f"Ordens executadas: {len(execute_orders)}")
logging.info(f"Capital utilizado: ${(optimal_positions.abs() * last_price.reindex(optimal_positions.index.str.replace('/USDC:USDC', ''), fill_value=1)).sum():.2f}")
