
import pandas as pd
import numpy as np



def format_from_database(formatted_df):
    """
    Reformats a structured DataFrame back to its original format,
    with unique 'Instrument' values as columns and 'Date' as the index.
    
    Parameters:
    formatted_df (pd.DataFrame): Input DataFrame with columns ['date', 'symbol', value_column].
    
    Returns:
    pd.DataFrame: Reformatted DataFrame with 'date' as index and symbols as columns, sorted by date.
    """
    
    # Identifying the value column (third column after 'date' and 'symbol')
    value_column = formatted_df.columns[2]
    
    # Pivot the DataFrame to reshape it into the original format
    original_df = formatted_df.pivot_table(
        index='date', columns='symbol', values=value_column, aggfunc='mean'
    )
    
    # Ensure datetime index and sort chronologically
    original_df.index = pd.to_datetime(original_df.index, errors='coerce')
    original_df = original_df.sort_index()
    
    return original_df


def calculate_vrm(returns, vol_span=35*96, smooth_span=10*96, long_term_window=365*96, quantile_window=365*96):
    """
    Calcula apenas o último valor do multiplicador de regime de volatilidade, de forma eficiente.
    """

    vol = returns.ewm(span=vol_span, min_periods=smooth_span).std().dropna()

    # Média de longo prazo
    long_term_avg = vol.rolling(window=long_term_window, min_periods=long_term_window).mean()

    # Vol normalizada suavizada
    norm_vol = (vol / long_term_avg).ewm(span=smooth_span).mean()
    
    last = norm_vol.iloc[-1]
    historical = norm_vol.iloc[:-1]
    
    vol_quantile = historical.apply(lambda x: np.sum(x < last[x.name]) / len(x.dropna()))
    
    
    multiplier = 2.0 - 1.5 * vol_quantile
    
    
    return multiplier


def calculate_idm(returns, risk_weights, lookback=365*96, max_idm=2.5):
    
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    valid_assets = risk_weights.index

    weights = risk_weights.values

    window_data = returns.tail(lookback)

    returns = window_data
    corr_matrix = returns.corr().values

    adjusted_corr_matrix = np.where(corr_matrix < 1, corr_matrix, 0)
    adjusted_corr_matrix = np.where(adjusted_corr_matrix < 0, 0, adjusted_corr_matrix)

    WHWT = weights.dot(adjusted_corr_matrix).dot(weights.transpose())

    diversification_multiplier = 1 / np.sqrt(WHWT) if WHWT != 0 else 0
   
    
    idm = min(diversification_multiplier, 2.5)
    
    return idm



def ewmac(price, Lfast, Lslow=None, vol_regime_multiplier=None):
    
    # Conversão inicial (float64 normalmente é mais performático para cálculos vetorizados)
    price = price.astype('float64')

    # Ajuste de janelas com base nos dados de 5 min
    Lfast_adjusted = round(Lfast * 96)
    Lslow = round(4 * Lfast_adjusted)
    vol_span = round(35 * 96)


    # Cálculo do EWMAC bruto
    fast_ewma = price.ewm(span=Lfast_adjusted).mean()
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma
    raw_ewmac = raw_ewmac.iloc[-1]

    # Volatility Adjustment
    stdev_returns = price.diff().ewm(span=vol_span).std() * np.sqrt((60/15) * 24 * 365)
    stdev_returns = stdev_returns.iloc[-1]

    # Forecast ajustado por volatilidade
    vol_adj_ewmac = raw_ewmac / stdev_returns
    vol_adj_ewmac.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Multiplicação elemento a elemento
    vol_adj_ewmac = vol_adj_ewmac * vol_regime_multiplier
    
    
    # Forecast scalar (ajuste empírico)
    scalar_dict = {16: 89.14, 8: 128.48, 4: 179.06, 2: 254.60}
    forecast_scalar = scalar_dict.get(Lfast)

    scaled_forecast = vol_adj_ewmac * forecast_scalar

    # Capar os forecasts em ±20
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast



def breakout(price, horizon, vol_regime_multiplier):
    price = price.astype('float64')

    # Ajuste da janela para frequência de 15 minutos (96 barras por dia)
    horizon_adjusted = round(horizon * 96)

    # Cálculo das bandas de breakout
    max_price = price.rolling(window=horizon_adjusted, min_periods=1).max()
    min_price = price.rolling(window=horizon_adjusted, min_periods=1).min()
    mean_price = (max_price + min_price) / 2

    # Evitar divisão por zero: (max - min) muito pequenos
    range_price = max_price - min_price
    range_price[range_price < 1e-6] = np.nan  # Protege contra divisão instável

    # Forecast bruto
    raw_forecast = 40 * (price - mean_price) / range_price

    # Suavização exponencial
    smoothed_forecast = raw_forecast.ewm(span=int(np.ceil(horizon_adjusted / 4))).mean()

    # Aplicar multiplicador de regime de volatilidade
        # Aplicar o multiplicador de regime de volatilidade
        
    smoothed_forecast = smoothed_forecast.iloc[-1]
    
    smoothed_forecast = smoothed_forecast * vol_regime_multiplier.astype('float64')

    # Scalar de forecast empírico
    scalar_dict = {40: 0.80, 20: 0.80, 10: 0.80, 5:0.80}
    forecast_scalar = scalar_dict.get(horizon)

    # Forecast escalado e capado
    scaled_forecast = smoothed_forecast * forecast_scalar
    capped_forecast = scaled_forecast.clip(lower=-20, upper=20)

    return capped_forecast
