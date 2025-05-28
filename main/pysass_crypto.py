import pandas as pd
import numpy as np
import gc

def format_from_database(formatted_df):
    """
    Reformats a structured DataFrame back to its original format.
    OTIMIZAÇÃO: Reduz uso de memória com dtypes específicos e processamento chunked.
    """
    # Otimização de tipos de dados
    if 'date' in formatted_df.columns:
        formatted_df['date'] = pd.to_datetime(formatted_df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # Identificar coluna de valor
    value_column = formatted_df.columns[2]
    
    # Converter para float32 para economizar memória (suficiente para preços crypto)
    formatted_df[value_column] = pd.to_numeric(formatted_df[value_column], errors='coerce').astype('float32')
    
    # Pivot otimizado
    original_df = formatted_df.pivot_table(
        index='date', 
        columns='symbol', 
        values=value_column, 
        aggfunc='last'  # 'last' é mais rápido que 'mean' para dados únicos
    )
    
    # Limpeza de memória
    del formatted_df
    gc.collect()
    
    # Garantir ordenação temporal
    original_df = original_df.sort_index()
    
    return original_df


def calculate_vrm(returns, vol_span=35*96, smooth_span=10*96, long_term_window=365*96, quantile_window=365*96):
    """
    OTIMIZAÇÃO: Calcula apenas o último valor VRM usando apenas dados necessários.
    """
    # Usar apenas dados suficientes para cálculo (economiza ~80% da memória)
    min_required = max(vol_span, long_term_window) + smooth_span
    if len(returns) > min_required:
        returns = returns.tail(min_required)
    
    # Volatilidade com float32
    vol = returns.ewm(span=vol_span, min_periods=vol_span//2).std().astype('float32')
    
    # Apenas últimas observações para média de longo prazo
    long_term_avg = vol.rolling(window=long_term_window, min_periods=long_term_window//2).mean()
    
    # Vol normalizada suavizada - apenas final
    norm_vol = (vol / long_term_avg).ewm(span=smooth_span).mean()
    
    # Otimização: calcular quantil apenas para o último valor
    last_val = norm_vol.iloc[-1]
    historical = norm_vol.iloc[:-1]
    
    # Quantil vetorizado mais eficiente
    vol_quantile = pd.Series(index=last_val.index, dtype='float32')
    for asset in last_val.index:
        if pd.notna(last_val[asset]) and len(historical[asset].dropna()) > 0:
            vol_quantile[asset] = (historical[asset] < last_val[asset]).sum() / len(historical[asset].dropna())
        else:
            vol_quantile[asset] = 0.5  # Default neutro
    
    multiplier = (2.0 - 1.5 * vol_quantile).clip(0.1, 3.0)  # Limitar range
    
    # Limpeza
    del vol, long_term_avg, norm_vol
    gc.collect()
    
    return multiplier


def calculate_idm(returns, risk_weights, lookback=365*96, max_idm=2.5):
    """
    OTIMIZAÇÃO: IDM mais eficiente com menos uso de memória.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Usar apenas dados necessários
    window_data = returns.tail(lookback)
    
    # Correlação apenas para ativos com pesos
    valid_assets = risk_weights.index
    returns_subset = window_data[valid_assets].dropna()
    
    if len(returns_subset) < 50:  # Mínimo de observações
        return 1.0
    
    # Matriz de correlação com float32
    corr_matrix = returns_subset.corr().values.astype('float32')
    
    # Ajustes de correlação mais eficientes
    np.fill_diagonal(corr_matrix, 1.0)  # Garantir diagonal = 1
    corr_matrix = np.clip(corr_matrix, 0, 1)  # Clip valores negativos
    
    weights = risk_weights.values.astype('float32')
    
    # Cálculo WHWT vetorizado
    WHWT = np.dot(weights, np.dot(corr_matrix, weights))
    
    # IDM com proteção contra divisão por zero
    if WHWT > 0:
        diversification_multiplier = min(1 / np.sqrt(WHWT), max_idm)
    else:
        diversification_multiplier = 1.0
    
    # Limpeza
    del corr_matrix, returns_subset
    gc.collect()
    
    return diversification_multiplier


def ewmac(price, Lfast, Lslow=None, vol_regime_multiplier=None):
    """
    OTIMIZAÇÃO: EWMAC mais eficiente usando apenas dados necessários.
    """
    # Ajuste de janelas
    Lfast_adjusted = round(Lfast * 96)
    Lslow = round(4 * Lfast_adjusted)
    vol_span = round(35 * 96)
    
    # Usar apenas dados suficientes (economiza ~70% da memória)
    min_required = max(Lslow, vol_span) * 3
    if len(price) > min_required:
        price_subset = price.tail(min_required).astype('float32')
    else:
        price_subset = price.astype('float32')
    
    # Cálculos EWMAC otimizados
    fast_ewma = price_subset.ewm(span=Lfast_adjusted, min_periods=Lfast_adjusted//2).mean()
    slow_ewma = price_subset.ewm(span=Lslow, min_periods=Lslow//2).mean()
    
    # Apenas último valor
    raw_ewmac = (fast_ewma - slow_ewma).iloc[-1]
    
    # Volatilidade ajustada
    returns = price_subset.diff()
    stdev_returns = returns.ewm(span=vol_span, min_periods=vol_span//2).std().iloc[-1]
    stdev_returns *= np.sqrt((60/15) * 24 * 365)
    
    # Forecast ajustado
    vol_adj_ewmac = raw_ewmac / stdev_returns
    vol_adj_ewmac = vol_adj_ewmac.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Aplicar multiplicador de regime
    if vol_regime_multiplier is not None:
        vol_adj_ewmac = vol_adj_ewmac * vol_regime_multiplier
    
    # Scalar e cap
    scalar_dict = {16: 89.14, 8: 128.48, 4: 179.06, 2: 254.60}
    forecast_scalar = scalar_dict.get(Lfast, 100)
    
    scaled_forecast = vol_adj_ewmac * forecast_scalar
    capped_forecast = scaled_forecast.clip(-20, 20)
    
    # Limpeza
    del fast_ewma, slow_ewma, returns, price_subset
    gc.collect()
    
    return capped_forecast


def breakout(price, horizon, vol_regime_multiplier):
    """
    OTIMIZAÇÃO: Breakout mais eficiente com menos uso de memória.
    """
    horizon_adjusted = round(horizon * 96)
    
    # Usar apenas dados necessários
    min_required = horizon_adjusted * 3
    if len(price) > min_required:
        price_subset = price.tail(min_required).astype('float32')
    else:
        price_subset = price.astype('float32')
    
    # Cálculos de breakout otimizados
    max_price = price_subset.rolling(window=horizon_adjusted, min_periods=horizon_adjusted//2).max()
    min_price = price_subset.rolling(window=horizon_adjusted, min_periods=horizon_adjusted//2).min()
    mean_price = (max_price + min_price) / 2
    
    # Proteção contra divisão por zero
    range_price = max_price - min_price
    range_price = range_price.where(range_price > 1e-6, np.nan)
    
    # Forecast bruto apenas para último valor
    current_price = price_subset.iloc[-1]
    current_mean = mean_price.iloc[-1]
    current_range = range_price.iloc[-1]
    
    raw_forecast = 40 * (current_price - current_mean) / current_range
    raw_forecast = raw_forecast.fillna(0)
    
    # Aplicar multiplicador de regime
    if vol_regime_multiplier is not None:
        raw_forecast = raw_forecast * vol_regime_multiplier.astype('float32')
    
    # Scalar e cap
    scalar_dict = {40: 0.80, 20: 0.80, 10: 0.80, 5: 0.80}
    forecast_scalar = scalar_dict.get(horizon, 0.80)
    
    scaled_forecast = raw_forecast * forecast_scalar
    capped_forecast = scaled_forecast.clip(-20, 20)
    
    # Limpeza
    del max_price, min_price, mean_price, range_price, price_subset
    gc.collect()
    
    return capped_forecast


def optimize_dataframe_memory(df):
    """
    Função auxiliar para otimizar uso de memória de DataFrames.
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    return df
