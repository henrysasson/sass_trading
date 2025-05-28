import pandas as pd
import numpy as np

def format_from_database(formatted_df):
    """
    Versão otimizada: usa pivot_table diretamente e tipos de dados eficientes
    """
    # Identificar coluna de valor
    value_column = formatted_df.columns[2]
    
    # Pivot otimizado com agregação explícita
    original_df = formatted_df.pivot_table(
        index='date', columns='symbol', values=value_column, 
        aggfunc='last', observed=True  # 'last' é mais eficiente que 'mean' para dados únicos
    )
    
    # Conversão eficiente de datetime
    if not pd.api.types.is_datetime64_any_dtype(original_df.index):
        original_df.index = pd.to_datetime(original_df.index, errors='coerce')
    
    return original_df.sort_index()

def calculate_vrm(returns, vol_span=35*96, smooth_span=10*96, long_term_window=365*96, quantile_window=365*96):
    """
    Versão otimizada: reduz uso de memória e cálculos desnecessários
    """
    # Usar apenas dados necessários
    min_required = max(vol_span, long_term_window) + smooth_span
    if len(returns) > min_required * 1.5:  # Se temos dados demais, usar apenas o necessário
        returns = returns.tail(int(min_required * 1.2))
    
    # Cálculo da volatilidade com tipos float32
    vol = returns.astype('float32').ewm(span=vol_span, min_periods=vol_span//2).std()
    vol = vol.dropna()
    
    # Média de longo prazo - apenas onde necessário
    long_term_avg = vol.rolling(window=long_term_window, min_periods=long_term_window//2).mean()
    
    # Vol normalizada suavizada
    norm_vol = (vol / long_term_avg).ewm(span=smooth_span).mean()
    
    # Cálculo eficiente do quantil apenas para o último valor
    last_values = norm_vol.iloc[-1]
    
    # Para cada ativo, calcular o quantil usando apenas dados históricos válidos
    vol_quantile = pd.Series(index=last_values.index, dtype='float32')
    
    for asset in last_values.index:
        historical_values = norm_vol[asset].iloc[:-1].dropna()
        if len(historical_values) > 10:  # Mínimo de pontos para quantil
            current_value = last_values[asset]
            vol_quantile[asset] = (historical_values < current_value).mean()
        else:
            vol_quantile[asset] = 0.5  # Default para casos com poucos dados
    
    # Multiplicador
    multiplier = 2.0 - 1.5 * vol_quantile
    
    return multiplier.astype('float32')

def calculate_idm(returns, risk_weights, lookback=365*96, max_idm=2.5):
    """
    Versão otimizada: usa menos memória e cálculos mais eficientes
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Usar apenas dados necessários
    window_data = returns.tail(lookback).astype('float32')
    
    # Calcular correlação de forma mais eficiente
    corr_matrix = window_data.corr().values.astype('float32')
    
    # Ajustar correlação (vetorizado)
    np.fill_diagonal(corr_matrix, 1.0)  # Garantir diagonal = 1
    corr_matrix = np.clip(corr_matrix, 0, 1)  # Clip é mais eficiente que where duplo
    
    # Cálculo IDM otimizado
    weights = risk_weights.values.astype('float32')
    WHWT = np.dot(weights, np.dot(corr_matrix, weights))
    
    if WHWT > 0:
        diversification_multiplier = 1 / np.sqrt(WHWT)
        idm = min(diversification_multiplier, max_idm)
    else:
        idm = 1.0
    
    return float(idm)

def ewmac(price, Lfast, Lslow=None, vol_regime_multiplier=None):
    """
    Versão otimizada: reduz cópias desnecessárias e usa cálculos vetorizados
    """
    # Trabalhar com float32 para economizar memória
    price_data = price.astype('float32')
    
    # Ajuste de janelas
    Lfast_adjusted = int(Lfast * 96)
    Lslow = int(4 * Lfast_adjusted)
    vol_span = int(35 * 96)
    
    # Usar apenas dados necessários (evitar cálculos desnecessários)
    min_required = max(Lslow, vol_span) * 2
    if len(price_data) > min_required:
        price_data = price_data.tail(min_required)
    
    # Cálculo EWMAC otimizado
    fast_ewma = price_data.ewm(span=Lfast_adjusted, min_periods=Lfast_adjusted//2).mean()
    slow_ewma = price_data.ewm(span=Lslow, min_periods=Lslow//2).mean()
    
    raw_ewmac = (fast_ewma - slow_ewma).iloc[-1]
    
    # Volatility Adjustment
    returns = price_data.diff()
    stdev_returns = returns.ewm(span=vol_span, min_periods=vol_span//2).std().iloc[-1]
    stdev_returns *= np.sqrt((60/15) * 24 * 365)
    
    # Forecast ajustado
    vol_adj_ewmac = raw_ewmac / stdev_returns
    vol_adj_ewmac = vol_adj_ewmac.replace([np.inf, -np.inf], 0)
    
    # Aplicar multiplicador de regime
    if vol_regime_multiplier is not None:
        vol_adj_ewmac = vol_adj_ewmac * vol_regime_multiplier.astype('float32')
    
    # Forecast scalar otimizado (lookup table)
    scalar_dict = {16: 89.14, 8: 128.48, 4: 179.06, 2: 254.60}
    forecast_scalar = scalar_dict.get(Lfast, 1.0)
    
    scaled_forecast = vol_adj_ewmac * forecast_scalar
    
    return scaled_forecast.clip(-20, 20).astype('float32')

def breakout(price, horizon, vol_regime_multiplier):
    """
    Versão otimizada: reduz uso de memória e cálculos redundantes
    """
    # Trabalhar com float32
    price_data = price.astype('float32')
    
    # Ajuste da janela
    horizon_adjusted = int(horizon * 96)
    
    # Usar apenas dados necessários
    min_required = horizon_adjusted * 3
    if len(price_data) > min_required:
        price_data = price_data.tail(min_required)
    
    # Cálculo otimizado das bandas
    rolling_window = price_data.rolling(window=horizon_adjusted, min_periods=max(1, horizon_adjusted//4))
    max_price = rolling_window.max()
    min_price = rolling_window.min()
    
    # Cálculos vetorizados
    mean_price = (max_price + min_price) * 0.5
    range_price = max_price - min_price
    
    # Proteção contra divisão por zero (mais eficiente)
    range_price = np.where(range_price < 1e-6, np.nan, range_price)
    
    # Forecast bruto
    raw_forecast = 40 * (price_data - mean_price) / range_price
    
    # Suavização
    smooth_span = max(1, horizon_adjusted // 4)
    smoothed_forecast = raw_forecast.ewm(span=smooth_span, min_periods=1).mean().iloc[-1]
    
    # Aplicar multiplicador de regime
    if vol_regime_multiplier is not None:
        smoothed_forecast = smoothed_forecast * vol_regime_multiplier.astype('float32')
    
    # Scalar (lookup table otimizada)
    scalar_dict = {40: 0.80, 20: 0.80, 10: 0.80, 5: 0.80}
    forecast_scalar = scalar_dict.get(horizon, 0.80)
    
    scaled_forecast = smoothed_forecast * forecast_scalar
    
    return scaled_forecast.clip(-20, 20).astype('float32')
