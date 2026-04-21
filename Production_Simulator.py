# Production_Simulator_enhanced.py
# Enhanced Production Simulator (improved)
# - SciPy optional with numpy fallback for simple exponential fit
# - Multiple outlier detection methods (zscore, iqr, rolling_mad)
# - Robust date parsing and duplicate handling
# - Logging of fit attempts and exportable run log
# - UI options for outlier method and SciPy install guidance
#
# Run with:
# streamlit run Production_Simulator_enhanced.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import io
import tempfile
import os
import math
import json
import logging
from dateutil import parser as date_parser

# Optional SciPy for curve fitting and stats
try:
    from scipy.optimize import curve_fit
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ---------------------------
# Logging setup
# ---------------------------
logger = logging.getLogger("prod_sim")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Stream handler for console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(sh)
    # File handler (rotating could be added)
    fh = logging.FileHandler("prod_sim_run.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

# Keep an in-memory run log for UI export
if 'run_log' not in st.session_state:
    st.session_state['run_log'] = []

def log_event(level, msg, **meta):
    entry = {"time": datetime.utcnow().isoformat() + "Z", "level": level, "message": msg, "meta": meta}
    st.session_state['run_log'].append(entry)
    if level == "INFO":
        logger.info(msg + " | " + json.dumps(meta))
    elif level == "WARNING":
        logger.warning(msg + " | " + json.dumps(meta))
    else:
        logger.error(msg + " | " + json.dumps(meta))

# ---------------------------
# Decline models
# ---------------------------
def arps_exponential(t, qi, di_annual_frac):
    di_monthly = di_annual_frac / 12.0
    return qi * np.exp(-di_monthly * t)

def arps_hyperbolic(t, qi, di_annual_frac, b):
    di_monthly = di_annual_frac / 12.0
    if np.isclose(b, 0.0):
        return qi * np.exp(-di_monthly * t)
    return qi / (1 + b * di_monthly * t)**(1.0 / b)

def duong_model(t, qi, m, tau):
    t_shift = np.maximum(t, 1.0)
    return qi * (t_shift ** (-m)) * np.exp(-tau * t_shift)

def modified_arps(t, qi, di_annual_frac, b, c):
    di_monthly = di_annual_frac / 12.0
    base = qi / (1 + b * di_monthly * t)**(1.0 / b) if not np.isclose(b, 0.0) else qi * np.exp(-di_monthly * t)
    return base * np.exp(-c * t)

# ---------------------------
# Robust date parsing and unit heuristics
# ---------------------------
def robust_parse_dates(series, dayfirst=False):
    parsed = pd.to_datetime(series, errors='coerce', dayfirst=dayfirst)
    mask = parsed.isna() & series.notna()
    for idx in series[mask].index:
        val = series.loc[idx]
        try:
            parsed.loc[idx] = date_parser.parse(str(val), dayfirst=dayfirst)
        except Exception:
            parsed.loc[idx] = pd.NaT
    return parsed

def detect_unit_suspect(values):
    # Heuristic: if median very small or very large, suggest unit mismatch
    vals = np.array([v for v in values if pd.notna(v)])
    if len(vals) == 0:
        return None
    med = np.median(vals)
    if med > 1e5:
        return "values unusually large (maybe units in liters or m3?)"
    if med < 1.0:
        return "values unusually small (maybe units in m3/d or normalized?)"
    return None

# ---------------------------
# Outlier detection utilities (multiple methods)
# ---------------------------
def detect_outliers_methods(df, rate_col='Oil_rate', method='zscore', **kwargs):
    series = df[rate_col].copy()
    mask = pd.Series(False, index=df.index)
    if method == 'zscore':
        try:
            from scipy import stats as _stats
            vals = series.fillna(method='ffill').fillna(method='bfill').values
            z = np.abs(_stats.zscore(vals))
            mask = pd.Series(z > kwargs.get('z_thresh', 3.0), index=df.index)
        except Exception:
            # fallback to numpy-based zscore
            vals = series.fillna(method='ffill').fillna(method='bfill').values
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if std == 0:
                mask = pd.Series(False, index=df.index)
            else:
                z = np.abs((vals - mean) / std)
                mask = pd.Series(z > kwargs.get('z_thresh', 3.0), index=df.index)
    elif method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - kwargs.get('k', 1.5) * iqr
        upper = q3 + kwargs.get('k', 1.5) * iqr
        mask = (series < lower) | (series > upper)
    elif method == 'rolling_mad':
        window = kwargs.get('window', 6)
        med = series.rolling(window, min_periods=1, center=True).median()
        mad = (series - med).abs().rolling(window, min_periods=1, center=True).median()
        mult = kwargs.get('mult', 3.5)
        mask = (series - med).abs() > mult * mad
    else:
        mask = pd.Series(False, index=df.index)
    return mask.fillna(False)

# ---------------------------
# Data cleaning utilities (extended)
# ---------------------------
def clean_production_df(df, date_col='Date', rate_col='Oil_rate', fill_method=None, outlier_method='zscore', outlier_params=None, dayfirst=False, aggregate_duplicates='mean'):
    """
    Clean production dataframe:
    - robust parse dates, sort
    - create MonthIndex (0-based)
    - remove or mark non-positive rates
    - reindex to continuous months and optionally fill missing values
    - detect outliers via chosen method and mark them (set to NaN)
    Returns: cleaned_df, report_dict, removed_nonpos_df, outliers_df
    """
    if outlier_params is None:
        outlier_params = {}
    df = df.copy()
    # Ensure columns exist
    if date_col not in df.columns:
        return pd.DataFrame(), {'error': f"Date column '{date_col}' not found."}, pd.DataFrame(), pd.DataFrame()
    if rate_col not in df.columns:
        df[rate_col] = np.nan

    # Parse dates robustly
    df[date_col] = robust_parse_dates(df[date_col], dayfirst=dayfirst)
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if df.empty:
        report = {'original_count': 0, 'final_points': 0, 'removed_nonpos': 0, 'outliers': 0, 'filled': 0}
        return pd.DataFrame(), report, pd.DataFrame(), pd.DataFrame()

    # Detect duplicates (same date) and aggregate
    dup_counts = df.duplicated(subset=[date_col], keep=False).sum()
    if dup_counts > 0:
        log_event("INFO", "Duplicate date rows detected", duplicates=int(dup_counts))
        if aggregate_duplicates == 'mean':
            df = df.groupby(date_col, as_index=False).agg({rate_col: 'mean'})
        elif aggregate_duplicates == 'sum':
            df = df.groupby(date_col, as_index=False).agg({rate_col: 'sum'})
        elif aggregate_duplicates == 'max':
            df = df.groupby(date_col, as_index=False).agg({rate_col: 'max'})
        else:
            df = df.drop_duplicates(subset=[date_col], keep='first')

    # Month index relative to first date
    df['MonthFloat'] = (df[date_col] - df[date_col].min()).dt.days / 30.44
    df['MonthIndex'] = np.round(df['MonthFloat']).astype(int)

    # Remove non-positive rates (mark them)
    removed_nonpos = pd.DataFrame()
    if rate_col in df.columns:
        removed_nonpos = df[df[rate_col] <= 0].copy()
        df = df[df[rate_col] > 0].copy()

    if df.empty:
        report = {'original_count': int(len(removed_nonpos)), 'final_points': 0, 'removed_nonpos': int(len(removed_nonpos)), 'outliers': 0, 'filled': 0}
        return pd.DataFrame(), report, removed_nonpos, pd.DataFrame()

    # Reindex to continuous months
    full_idx = np.arange(df['MonthIndex'].min(), df['MonthIndex'].max() + 1)
    df_full = pd.DataFrame({'MonthIndex': full_idx})
    df = df_full.merge(df[['MonthIndex', rate_col, date_col]], on='MonthIndex', how='left')

    # Optionally fill missing rates
    filled_count = 0
    if fill_method == 'linear':
        before_na = df[rate_col].isna().sum()
        df[rate_col] = df[rate_col].interpolate(method='linear', limit_direction='both')
        filled_count = before_na - df[rate_col].isna().sum()
    elif fill_method == 'ffill':
        before_na = df[rate_col].isna().sum()
        df[rate_col] = df[rate_col].fillna(method='ffill').fillna(method='bfill')
        filled_count = before_na - df[rate_col].isna().sum()

    # Outlier detection using chosen method
    outliers = pd.DataFrame()
    if df[rate_col].notna().sum() >= 3:
        try:
            mask = detect_outliers_methods(df, rate_col=rate_col, method=outlier_method, **outlier_params)
            outliers = df.loc[mask, :].copy()
            df.loc[mask, rate_col] = np.nan
            log_event("INFO", "Outlier detection applied", method=outlier_method, outliers=int(mask.sum()))
        except Exception as e:
            log_event("WARNING", "Outlier detection failed", error=str(e))
            outliers = pd.DataFrame()

    report = {
        'original_count': int(df_full.shape[0] + len(removed_nonpos)),
        'final_points': int(df[rate_col].notna().sum()),
        'removed_nonpos': int(len(removed_nonpos)),
        'outliers': int(len(outliers)),
        'filled': int(filled_count),
        'month_start': int(full_idx.min()),
        'month_end': int(full_idx.max())
    }
    # Unit suspicion
    unit_note = detect_unit_suspect(df[rate_col].dropna().values)
    if unit_note:
        report['unit_suspect'] = unit_note
        log_event("WARNING", "Unit suspicion detected", note=unit_note)
    return df, report, removed_nonpos, outliers

# ---------------------------
# Fit metrics & retries (with SciPy fallback)
# ---------------------------
def compute_fit_metrics(y, y_pred, k):
    n = len(y)
    sse = float(np.sum((y - y_pred)**2))
    rmse = float(np.sqrt(sse / n)) if n > 0 else float('nan')
    sigma2 = sse / n if sse > 0 and n > 0 else 1e-9
    aic = float(2 * k + n * np.log(sigma2))
    bic = float(k * np.log(n) + n * np.log(sigma2)) if n > 0 else float('nan')
    return {'sse': sse, 'rmse': rmse, 'aic': aic, 'bic': bic}

def try_curve_fit_with_retries(func, x, y, p0_list, bounds_list, maxfev=20000):
    last_err = None
    if not SCIPY_AVAILABLE:
        return None, None, "SciPy not available"
    for p0 in p0_list:
        for bounds in bounds_list:
            try:
                log_event("INFO", "Attempting curve_fit", p0=p0, bounds=bounds)
                popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
                log_event("INFO", "curve_fit succeeded", popt=list(map(float, popt)))
                return popt, pcov, None
            except Exception as e:
                last_err = e
                log_event("WARNING", "curve_fit attempt failed", error=str(e), p0=p0, bounds=bounds)
    return None, None, str(last_err)

# Simple numpy-based exponential fit fallback
def simple_exponential_fit(x, y):
    mask = (~np.isnan(y)) & (y > 0)
    if mask.sum() < 3:
        raise ValueError("Insufficient data for simple fit")
    x_fit = x[mask]
    y_fit = y[mask]
    ln_y = np.log(y_fit)
    A = np.vstack([np.ones_like(x_fit), -x_fit]).T
    coeffs, *_ = np.linalg.lstsq(A, ln_y, rcond=None)
    ln_qi, d = coeffs[0], coeffs[1]
    qi = float(np.exp(ln_qi))
    di_annual_frac = float(d * 12.0)
    log_event("INFO", "simple_exponential_fit used", qi=qi, di_annual_frac=di_annual_frac)
    return {'qi': qi, 'di_annual_frac': di_annual_frac}

def fit_decline_models(months_arr, rates_arr, try_retries=True):
    """
    Fit multiple decline models and return params + metrics.
    Uses SciPy if available; otherwise uses simple fallback for exponential only.
    """
    results = {}
    x = np.array(months_arr, dtype=float)
    y = np.array(rates_arr, dtype=float)
    mask = ~np.isnan(y) & (y > 0)
    if mask.sum() < 3:
        return {"error": "Insufficient positive data points for fitting."}
    x_fit = x[mask]
    y_fit = y[mask]

    # Exponential
    try:
        if SCIPY_AVAILABLE:
            p0_list = [[float(y_fit[0]), 0.2], [float(np.mean(y_fit)), 0.1]]
            bounds_list = [([0, 0], [np.inf, 1.0])]
            popt, pcov, err = try_curve_fit_with_retries(lambda t, qi, di: arps_exponential(t, qi, di), x_fit, y_fit, p0_list, bounds_list, maxfev=10000)
            if err:
                raise Exception(err)
            pred = arps_exponential(x_fit, *popt)
            metrics = compute_fit_metrics(y_fit, pred, k=2)
            results['Exponential'] = {'params': {'qi': float(popt[0]), 'di_annual_frac': float(popt[1])}, 'metrics': metrics}
        else:
            # fallback
            params = simple_exponential_fit(x_fit, y_fit)
            pred = arps_exponential(x_fit, params['qi'], params['di_annual_frac'])
            metrics = compute_fit_metrics(y_fit, pred, k=2)
            results['Exponential'] = {'params': params, 'metrics': metrics, 'note': 'fallback_numpy'}
    except Exception as e:
        results['Exponential'] = {'error': str(e)}
        log_event("ERROR", "Exponential fit failed", error=str(e))

    # Hyperbolic, Duong, ModifiedArps only if SciPy available
    if SCIPY_AVAILABLE:
        # Hyperbolic
        try:
            p0_list = [[float(y_fit[0]), 0.2, 0.5], [float(np.mean(y_fit)), 0.15, 0.3]]
            bounds_list = [([0, 0, 0], [np.inf, 1.0, 2.0])]
            popt, pcov, err = try_curve_fit_with_retries(lambda t, qi, di, b: arps_hyperbolic(t, qi, di, b), x_fit, y_fit, p0_list, bounds_list, maxfev=20000)
            if err:
                raise Exception(err)
            pred = arps_hyperbolic(x_fit, *popt)
            metrics = compute_fit_metrics(y_fit, pred, k=3)
            results['Hyperbolic'] = {'params': {'qi': float(popt[0]), 'di_annual_frac': float(popt[1]), 'b': float(popt[2])}, 'metrics': metrics}
        except Exception as e:
            results['Hyperbolic'] = {'error': str(e)}
            log_event("WARNING", "Hyperbolic fit failed", error=str(e))

        # Duong
        try:
            p0_list = [[float(y_fit[0]), 0.1, 0.001], [float(np.mean(y_fit)), 0.05, 0.0005]]
            bounds_list = [([0, 0, 0], [np.inf, 1.0, 1.0])]
            popt, pcov, err = try_curve_fit_with_retries(lambda t, qi, m, tau: duong_model(t, qi, m, tau), x_fit + 1.0, y_fit, p0_list, bounds_list, maxfev=20000)
            if err:
                raise Exception(err)
            pred = duong_model(x_fit + 1.0, *popt)
            metrics = compute_fit_metrics(y_fit, pred, k=3)
            results['Duong'] = {'params': {'qi': float(popt[0]), 'm': float(popt[1]), 'tau': float(popt[2])}, 'metrics': metrics}
        except Exception as e:
            results['Duong'] = {'error': str(e)}
            log_event("WARNING", "Duong fit failed", error=str(e))

        # Modified Arps
        try:
            p0_list = [[float(y_fit[0]), 0.2, 0.5, 0.001], [float(np.mean(y_fit)), 0.15, 0.3, 0.0005]]
            bounds_list = [([0, 0, 0, 0], [np.inf, 1.0, 2.0, 1.0])]
            popt, pcov, err = try_curve_fit_with_retries(lambda t, qi, di, b, c: modified_arps(t, qi, di, b, c), x_fit, y_fit, p0_list, bounds_list, maxfev=30000)
            if err:
                raise Exception(err)
            pred = modified_arps(x_fit, *popt)
            metrics = compute_fit_metrics(y_fit, pred, k=4)
            results['ModifiedArps'] = {'params': {'qi': float(popt[0]), 'di_annual_frac': float(popt[1]), 'b': float(popt[2]), 'c': float(popt[3])}, 'metrics': metrics}
        except Exception as e:
            results['ModifiedArps'] = {'error': str(e)}
            log_event("WARNING", "ModifiedArps fit failed", error=str(e))
    else:
        log_event("INFO", "SciPy not available; only exponential fallback performed")

    # Best model by AIC
    best = None
    best_aic = np.inf
    for name, res in results.items():
        if isinstance(res, dict) and 'metrics' in res:
            aic = res['metrics'].get('aic', np.inf)
            if aic < best_aic:
                best_aic = aic
                best = name
    results['best_model'] = best
    log_event("INFO", "Fitting completed", best_model=best)
    return results

# ---------------------------
# Economics & simulation utilities
# ---------------------------
def simulate_rates_from_model(model_name, params, months):
    t = np.arange(months)
    if model_name == 'Exponential':
        return arps_exponential(t, params['qi'], params['di_annual_frac'])
    if model_name == 'Hyperbolic':
        return arps_hyperbolic(t, params['qi'], params['di_annual_frac'], params['b'])
    if model_name == 'Duong':
        return duong_model(t + 1.0, params['qi'], params['m'], params['tau'])
    if model_name == 'ModifiedArps':
        return modified_arps(t, params['qi'], params['di_annual_frac'], params['b'], params['c'])
    return np.zeros(months)

def calculate_economics(rates, oil_price, opex_fixed, discount_annual=0.10):
    days_in_month = 30.44
    monthly_prod = rates * days_in_month
    revenue = monthly_prod * oil_price
    profit = revenue - opex_fixed
    cum_prod = np.cumsum(monthly_prod)
    discount_rate = discount_annual / 12.0
    discounts = (1 + discount_rate) ** np.arange(len(rates))
    npv = float(np.sum(profit / discounts))
    return cum_prod, profit, npv

def compute_payout_month(cum_profit, capex):
    if capex <= 0:
        return 0
    cum = np.cumsum(cum_profit)
    idx = np.where(cum >= capex)[0]
    if len(idx) == 0:
        return None
    return int(idx[0]) + 1

# ---------------------------
# Insights (unchanged)
# ---------------------------
def compute_confidence(n_points, rmse, aic, pct_filled):
    score = 0.0
    score += min(1.0, n_points / 12.0) * 0.45
    rmse_score = 1.0 / (1.0 + math.log1p(1 + rmse)) if not math.isnan(rmse) else 0.0
    score += rmse_score * 0.30
    aic_score = 1.0 / (1.0 + abs(aic) / 1000.0) if not math.isnan(aic) else 0.0
    score += aic_score * 0.15
    score += min(1.0, pct_filled) * 0.10
    return max(0.0, min(1.0, score))

def make_insight(type_key, confidence, evidence, recommendation, explain):
    return {
        "type": type_key,
        "confidence": float(round(confidence, 3)),
        "evidence": evidence,
        "recommendation": recommendation,
        "explain": explain
    }

def generate_insights_structured(fit_results, best_model_params, cleaned_df, econ=None):
    insights = []
    n_points = int(cleaned_df.dropna().shape[0]) if cleaned_df is not None else 0
    pct_filled = 0.0
    if cleaned_df is not None:
        total_months = int(cleaned_df.shape[0])
        if total_months > 0:
            pct_filled = float(cleaned_df['Oil_rate'].notna().sum()) / total_months

    best = fit_results.get('best_model') if isinstance(fit_results, dict) else None
    if best:
        insights.append(make_insight(
            "best_model",
            0.9,
            {"model": best},
            f"Model {best} best explains historical rates (by AIC).",
            f"Model selection based on AIC across fitted models."
        ))

    metrics = None
    if best and best in fit_results and 'metrics' in fit_results[best]:
        metrics = fit_results[best]['metrics']
    else:
        for k, v in fit_results.items():
            if isinstance(v, dict) and 'metrics' in v:
                metrics = v['metrics']
                break

    rmse = metrics.get('rmse') if metrics else float('nan')
    aic = metrics.get('aic') if metrics else float('nan')
    confidence = compute_confidence(n_points, rmse if not math.isnan(rmse) else 1e6, aic if not math.isnan(aic) else 1e6, pct_filled)

    if best_model_params and 'qi' in best_model_params:
        qi = best_model_params.get('qi')
        recent_peak = float(np.nanmax(cleaned_df['Oil_rate'].fillna(0))) if cleaned_df is not None and 'Oil_rate' in cleaned_df.columns else qi
        if qi > 0 and recent_peak / qi < 0.8:
            insights.append(make_insight(
                "early_drop_vs_qi",
                0.8,
                {"qi": qi, "recent_peak": recent_peak, "ratio": round(recent_peak/qi, 3)},
                "تحقق من جودة القياسات وعمليات السطح (chokes, workovers).",
                "الفرق الكبير بين Qi والنقطة التاريخية العليا قد يشير إلى قياسات غير متسقة أو تغييرات تشغيلية."
            ))

    if 'Hyperbolic' in fit_results and isinstance(fit_results['Hyperbolic'], dict) and 'params' in fit_results['Hyperbolic']:
        b = fit_results['Hyperbolic']['params'].get('b', None)
        if b is not None:
            if b < 0.3:
                insights.append(make_insight(
                    "b_low_steep_decline",
                    0.85,
                    {"b": b},
                    "راجع استراتيجيات Artificial Lift وفحص ضغط الخزان.",
                    "B < 0.3 عادةً يدل على هبوط حاد؛ قد يكون السبب استنزاف ضغط الخزان أو قيود سطحية."
                ))
            elif b > 1.0:
                insights.append(make_insight(
                    "b_high_slow_decline",
                    0.7,
                    {"b": b},
                    "قد لا تحتاج لتدخل فوري؛ راجع دعم الخزان وبيانات الضغط.",
                    "B > 1 يشير لانحدار أبطأ، قد يكون الخزان ذو دعم جيد."
                ))

    if 'Duong' in fit_results and isinstance(fit_results['Duong'], dict) and 'params' in fit_results['Duong']:
        m = fit_results['Duong']['params'].get('m', None)
        if m is not None and m > 0.5:
            insights.append(make_insight(
                "duong_m_high",
                0.65,
                {"m": m},
                "راجع بيانات الضغط وعمليات الحقن إن وُجدت.",
                "m مرتفع في نموذج Duong قد يشير لسلوك انتقالي قوي."
            ))

    if econ and 'npv' in econ:
        npv = econ.get('npv')
        if npv is not None and npv < 0:
            insights.append(make_insight(
                "npv_negative",
                0.9,
                {"npv": npv},
                "راجع فروض السعر وOPEX وCAPEX أو فكر في تحسين الإنتاج/خفض التكاليف.",
                "NPV سالب يشير إلى أن المشروع قد لا يكون مربحاً تحت الفروض الحالية."
            ))

    if len(insights) == 0:
        insights.append(make_insight(
            "no_critical_findings",
            confidence,
            {"n_points": n_points, "rmse": rmse, "aic": aic},
            "الملاءمة تمت بنجاح؛ راجع الرسوم التفصيلية لاختيار النموذج النهائي.",
            "لا توجد ملاحظات حرجة تلقائية بناءً على القواعد الحالية."
        ))

    exec_summary = f"Generated {len(insights)} insight(s). Confidence composite: {round(confidence,2)}."
    return insights, exec_summary

# ---------------------------
# Sensitivity analysis
# ---------------------------
def run_price_sensitivity(base_rates, price_scenarios, opex_scenarios, capex, discount_annual=0.10):
    rows = []
    months = len(base_rates)
    for price in price_scenarios:
        for opex in opex_scenarios:
            cum, profit, npv = calculate_economics(base_rates, price, opex, discount_annual=discount_annual)
            payout = compute_payout_month(profit, capex)
            rows.append({'price': float(price), 'opex': float(opex), 'npv': float(npv), 'payout': payout})
    return pd.DataFrame(rows)

# ---------------------------
# Session state initialization
# ---------------------------
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}
if 'imported_data' not in st.session_state:
    st.session_state.imported_data = None
if 'fitted_models' not in st.session_state:
    st.session_state.fitted_models = {}
if 'mapping' not in st.session_state:
    st.session_state.mapping = None

# ==========================================
# Sidebar branding (kept) + SciPy guidance
# ==========================================
petroscience_logo_url = "https://media.licdn.com/dms/image/v2/D5603AQGFR4yZ0Xbu8g/profile-displayphoto-scale_400_400/B56Zxrp.xPKgAg-/0/1771332696780?e=1778112000&v=beta&t=tAmGd--8fgLzeWLXKFNGWQDkBBvcV-zxaRUUekQjEA0"
st.sidebar.image(petroscience_logo_url, use_container_width=True)
st.sidebar.markdown("---")
my_photo_url = "https://media.licdn.com/dms/image/v2/D4D03AQH_gUWhtKDArA/profile-displayphoto-scale_400_400/B4DZxtywF.HwAg-/0/1771368594755?e=1778112000&v=beta&t=2zHjZvkL_46hw9zD8S40rnYdzkWgSINDA7aLZ_f8k8U"
col1, col2 = st.sidebar.columns([1, 3])
with col1:
    st.image(my_photo_url, width=60)
with col2:
    st.write("### المهندس عمار أسعد")
    st.write("مطور التطبيق")
st.sidebar.markdown("---")
st.sidebar.title("📩 تواصل معي")
linkedin_url = "https://www.linkedin.com/in/ammar-asaad/"
st.sidebar.markdown(f'[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)]({linkedin_url})')
email = "ammarasaad727@gmail.com"
st.sidebar.write(f"📧: {email}")
st.sidebar.markdown("---")
st.sidebar.info("تم تطوير هذا التطبيق بواسطة المهندس عمار أسعد")

# SciPy status and guidance
st.sidebar.markdown("### SciPy status")
if SCIPY_AVAILABLE:
    st.sidebar.success("SciPy متوفر — الملاءمة الكاملة ممكنة.")
else:
    st.sidebar.warning("SciPy غير مثبت — سيتم استخدام ملاءمة مبسطة (exponential) كـ fallback.")
    st.sidebar.code("pip install scipy")
    if st.sidebar.button("نسخ أمر التثبيت"):
        st.sidebar.write("pip install scipy")  # simple feedback

# ---------------------------
# Import UI and Column Mapping
# ---------------------------
st.subheader("📥 استيراد بيانات الإنتاج (CSV / Excel)")
uploaded = st.file_uploader("ارفع ملف إنتاج (يمكن أن يحتوي أي أسماء أعمدة) — استخدم Column Mapping", type=['csv', 'xlsx', 'xls'], key="prod_upload")
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(('.xls', '.xlsx')):
            df_in = pd.read_excel(uploaded)
        else:
            df_in = pd.read_csv(uploaded)
        df_in.columns = [c.strip() for c in df_in.columns]
        st.session_state.imported_data = df_in
        st.success("تم استيراد الملف. قم بتعيين الـ Column Mapping أدناه.")
        log_event("INFO", "File imported", filename=uploaded.name, ncols=len(df_in.columns))
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {e}")
        log_event("ERROR", "File import failed", error=str(e))

if st.session_state.imported_data is not None:
    st.markdown("**🧭 Column Mapping (اجعل الملف Universal)**")
    cols = list(st.session_state.imported_data.columns)
    def guess_column(candidates, targets):
        for t in targets:
            for c in candidates:
                if t.lower() in c.lower():
                    return c
        return None
    default_well = guess_column(cols, ['well', 'well_id', 'wellid', 'well name', 'wellid'])
    default_date = guess_column(cols, ['date', 'time', 'day'])
    default_rate = guess_column(cols, ['oil_rate', 'rate', 'production', 'oil', 'oilrate'])
    mapped_well = st.selectbox("عمود معرف البئر (Well ID):", options=[None] + cols, index=cols.index(default_well) + 1 if default_well in cols else 0)
    mapped_date = st.selectbox("عمود التاريخ (Date):", options=[None] + cols, index=cols.index(default_date) + 1 if default_date in cols else 0)
    mapped_rate = st.selectbox("عمود الإنتاج (Oil rate):", options=[None] + cols, index=cols.index(default_rate) + 1 if default_rate in cols else 0)
    mapping_confirm = st.button("تأكيد الـ Mapping")
    if mapping_confirm:
        if not mapped_well or not mapped_date or not mapped_rate:
            st.error("الرجاء تعيين جميع الأعمدة الثلاثة: Well ID, Date, Oil rate.")
        else:
            st.session_state.mapping = {'well': mapped_well, 'date': mapped_date, 'rate': mapped_rate}
            st.success("تم حفظ الـ Column Mapping. يمكنك الآن اختيار بئر للملاءمة أو إنشاء سيناريو يدوي.")
            log_event("INFO", "Column mapping saved", mapping=st.session_state.mapping)

# ---------------------------
# Manual scenario UI (unchanged)
# ---------------------------
st.markdown("---")
st.subheader("🛠️ إعدادات سيناريو يدوي")
name = st.text_input("اسم السيناريو (يدوي):", value=f"Scenario {len(st.session_state.scenarios)+1}", key="manual_name")
col_a, col_b = st.columns(2)
qi = col_a.number_input("Qi (STB/D):", min_value=0.0, value=500.0, step=10.0, format="%.2f", key="manual_qi")
di_input_pct = col_b.number_input("Annual Di (%):", min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.2f", key="manual_di")
di_annual = float(di_input_pct) / 100.0
b_factor = st.slider("B-Factor (Arps):", 0.0, 2.0, 0.5, step=0.01, key="manual_b")
months = int(st.number_input("المدة (أشهر):", min_value=1, max_value=1200, value=36, step=1, key="manual_months"))
st.markdown("---")
st.subheader("💰 بارامترات اقتصادية")
oil_price = st.number_input("سعر البرميل ($):", min_value=0.0, value=75.0, step=0.5, format="%.2f", key="manual_price")
opex = st.number_input("مصاريف التشغيل ($/شهر):", min_value=0.0, value=5000.0, step=100.0, format="%.2f", key="manual_opex")
capex = st.number_input("CAPEX (تكلفة البئر $):", min_value=0.0, value=0.0, step=1000.0, format="%.2f", key="manual_capex")
discount_rate = st.number_input("معدل الخصم السنوي (%):", min_value=0.0, max_value=100.0, value=10.0, step=0.1) / 100.0
st.markdown("---")
st.subheader("⚙️ نموذج الانحدار")
model_choice = st.selectbox("اختر نموذج الانحدار:", ['Exponential', 'Hyperbolic', 'Duong', 'ModifiedArps'], key="manual_model")
if st.button("➕ حفظ السيناريو اليدوي", type="primary", key="save_manual"):
    params = {}
    if model_choice == 'Exponential':
        params = {'qi': float(qi), 'di_annual_frac': float(di_annual)}
    elif model_choice == 'Hyperbolic':
        params = {'qi': float(qi), 'di_annual_frac': float(di_annual), 'b': float(b_factor)}
    elif model_choice == 'Duong':
        params = {'qi': float(qi), 'm': 0.1, 'tau': 0.001}
    elif model_choice == 'ModifiedArps':
        params = {'qi': float(qi), 'di_annual_frac': float(di_annual), 'b': float(b_factor), 'c': 0.001}
    rates = simulate_rates_from_model(model_choice, params, months)
    cum, profit, npv = calculate_economics(rates, oil_price, opex, discount_annual=discount_rate)
    payout = compute_payout_month(profit, capex) if capex > 0 else 0
    st.session_state.scenarios[name] = {
        'rates': rates,
        'cum': cum,
        'profit': profit,
        'npv': npv,
        'peak': float(np.max(rates)),
        'params': params,
        'econ': {'oil_price': float(oil_price), 'opex': float(opex), 'capex': float(capex), 'discount_annual': float(discount_rate)},
        'model': model_choice,
        'payout_month': payout
    }
    st.success(f"تم حفظ السيناريو: {name}")
    log_event("INFO", "Manual scenario saved", name=name, model=model_choice)

# ---------------------------
# Main UI Tabs (Fitting tab includes outlier options and fit button)
# ---------------------------
st.title("📊 Production Scenario Manager & Simulator — Petro Science")
tabs = st.tabs(["📈 تحليل السيناريوهات", "⚖️ المقارنة", "💾 تصدير", "🔧 ملاءمة (Fitting)", "🔬 Insights", "🔬 حساسية"])

with tabs[0]:
    if not st.session_state.scenarios:
        st.info("قم بإضافة سيناريو من الشريط الجانبي أو استورد بيانات تاريخية للبدء.")
    else:
        selected_sc = st.selectbox("اختر سيناريو للعرض التفصيلي:", list(st.session_state.scenarios.keys()), key="main_select")
        data = st.session_state.scenarios[selected_sc]
        c1, c2, c3, c4 = st.columns(4)
        try:
            c1.metric("الإنتاج التراكمي (STB)", f"{int(data['cum'][-1]):,}")
        except Exception:
            c1.metric("الإنتاج التراكمي (STB)", "N/A")
        c2.metric("صافي القيمة الحالية (NPV)", f"${int(data['npv']):,}")
        c3.metric("أعلى إنتاج (Peak)", f"{int(data['peak']):,} BPD")
        payout_text = f"شهر {data['payout_month']}" if data.get('payout_month') else ("لا يوجد CAPEX" if data.get('econ', {}).get('capex', 0) == 0 else "لم يتحقق بعد")
        c4.metric("فترة الاسترداد (Payout)", payout_text)

        months_range = np.arange(len(data['rates'])) + 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months_range, y=list(data['rates']), name="Production Rate (BPD)"))
        fig.add_trace(go.Scatter(x=months_range, y=list(data['cum']), name="Cumulative (STB)", yaxis="y2"))
        fig.update_layout(title=f"منحنى الإنتاج والكمية التراكمية: {selected_sc}", xaxis_title="شهر", yaxis=dict(title="BPD"), yaxis2=dict(title="Cumulative STB", overlaying="y", side="right"), template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)

        df_preview = pd.DataFrame({
            "Month": months_range,
            "Rate (BPD)": np.round(pd.Series(data['rates']), 2),
            "Cumulative (STB)": np.round(pd.Series(data['cum']), 2),
            "Profit ($/month)": np.round(pd.Series(data['profit']), 2)
        })
        st.markdown("#### جدول ملخّص (أول 24 شهرًا)")
        st.dataframe(df_preview.head(24), use_container_width=True)

with tabs[1]:
    if len(st.session_state.scenarios) > 1:
        st.subheader("⚖️ مقارنة الأداء بين السيناريوهات")
        comp_fig = go.Figure()
        comparison_data = []
        for sc_name, sc_data in st.session_state.scenarios.items():
            comp_fig.add_trace(go.Scatter(y=list(sc_data['rates']), name=sc_name))
            comparison_data.append({
                "Scenario": sc_name,
                "Model": sc_data.get('model', 'Custom'),
                "NPV ($)": f"{sc_data['npv']:,.0f}",
                "Cum Prod (bbl)": f"{sc_data['cum'][-1]:,.0f}",
                "Qi (bpd)": f"{sc_data['params'].get('qi', np.nan):,.0f}"
            })
        comp_fig.update_layout(title="مقارنة معدلات الإنتاج", xaxis_title="الشهر", yaxis_title="BPD", template="plotly_white")
        st.plotly_chart(comp_fig, use_container_width=True)
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

with tabs[3]:
    st.header("🔧 ملاءمة (Fitting)")
    if st.session_state.imported_data is None or st.session_state.mapping is None:
        st.info("استورد بيانات وحدد Column Mapping للبدء بالملاءمة.")
    else:
        mapping = st.session_state.mapping
        df_raw = st.session_state.imported_data.copy()
        well_col = mapping['well']
        date_col = mapping['date']
        rate_col = mapping['rate']
        wells = df_raw[well_col].dropna().unique().tolist()
        selected_well = st.selectbox("اختر بئراً للملاءمة:", options=wells)
        outlier_method = st.selectbox("طريقة كشف الشذوذ:", ['zscore', 'iqr', 'rolling_mad'])
        z_thresh = st.number_input("z-score threshold:", value=3.0, step=0.1) if outlier_method == 'zscore' else None
        iqr_k = st.number_input("IQR multiplier k:", value=1.5, step=0.1) if outlier_method == 'iqr' else None
        rolling_window = st.number_input("Rolling window (months):", value=6, step=1) if outlier_method == 'rolling_mad' else None
        dayfirst = st.checkbox("Parse dates with dayfirst", value=False)
        aggregate_duplicates = st.selectbox("Aggregate duplicate dates by:", ['mean', 'sum', 'max', 'first'])
        if st.button("تشغيل الملاءمة على البئر المحدد"):
            df_well = df_raw[df_raw[well_col] == selected_well].copy()
            df_well[date_col] = robust_parse_dates(df_well[date_col], dayfirst=dayfirst)
            df_well = df_well.sort_values(date_col).reset_index(drop=True)
            df_well = df_well.rename(columns={date_col: 'Date', rate_col: 'Oil_rate'})
            cleaned_df, report, removed_nonpos, outliers = clean_production_df(df_well, date_col='Date', rate_col='Oil_rate', fill_method='linear', outlier_method=outlier_method, outlier_params={'z_thresh': z_thresh, 'k': iqr_k, 'window': rolling_window, 'mult': 3.5}, dayfirst=dayfirst, aggregate_duplicates=aggregate_duplicates)
            st.write("تنبيه تنظيف البيانات:", report)
            if not cleaned_df.empty:
                months_arr = cleaned_df['MonthIndex'].values
                rates_arr = cleaned_df['Oil_rate'].values
                fit_results = fit_decline_models(months_arr, rates_arr)
                st.session_state.fitted_models[selected_well] = {'fit_results': fit_results, 'cleaned_df': cleaned_df, 'report': report}
                st.success("انتهت الملاءمة. راجع النتائج في تبويب Insights.")
                log_event("INFO", "Fitting run completed", well=selected_well, report=report, fit_summary={'best_model': fit_results.get('best_model')})
            else:
                st.error("لا توجد بيانات كافية بعد التنظيف للملاءمة.")
                log_event("WARNING", "No cleaned data for fitting", well=selected_well, report=report)

with tabs[4]:
    st.header("🔬 Insights")
    if not st.session_state.fitted_models:
        st.info("لا توجد نتائج ملاءمة بعد. نفّذ ملاءمة أولاً.")
    else:
        well_list = list(st.session_state.fitted_models.keys())
        sel = st.selectbox("اختر بئراً لعرض Insights:", well_list)
        fm = st.session_state.fitted_models[sel]
        
        # التعديل هنا: استخدام .get() للوصول الآمن للبيانات
        fit_results = fm.get('fit_results', {})
        cleaned_df = fm.get('cleaned_df', None)
        
        best = fit_results.get('best_model')
        best_params = None
        if best and best in fit_results and 'params' in fit_results[best]:
            best_params = fit_results[best]['params']
            
        # الدالة generate_insights_structured مصممة مسبقاً للتعامل مع cleaned_df إذا كان None
        insights, summary = generate_insights_structured(fit_results, best_params, cleaned_df, econ=None)
        
        st.markdown("**Executive Summary**")
        st.write(summary)
        st.markdown("**Insights**")
        for ins in insights:
            st.json(ins)

with tabs[2]:
    st.header("💾 تصدير")
    st.markdown("### سجل التشغيل (Run Log)")
    st.write(f"Entries: {len(st.session_state['run_log'])}")
    if st.button("عرض سجل التشغيل"):
        st.json(st.session_state['run_log'])
    if st.button("تصدير سجل التشغيل كـ JSON"):
        b = json.dumps(st.session_state['run_log'], indent=2).encode('utf-8')
        st.download_button("Download run_log.json", data=b, file_name="run_log.json", mime="application/json")
    st.markdown("---")
    st.markdown("### تصدير السيناريوهات")
    if st.session_state.scenarios:
        if st.button("تصدير جميع السيناريوهات كـ JSON"):
            b = json.dumps(st.session_state.scenarios, default=lambda o: (o.tolist() if isinstance(o, np.ndarray) else str(o)), indent=2).encode('utf-8')
            st.download_button("Download scenarios.json", data=b, file_name="scenarios.json", mime="application/json")
    else:
        st.info("لا توجد سيناريوهات للتصدير.")

with tabs[5]:
    st.header("🔬 حساسية")
    if st.session_state.scenarios:
        sc_name = st.selectbox("اختر سيناريو للحساسية:", list(st.session_state.scenarios.keys()))
        sc = st.session_state.scenarios[sc_name]
        base_rates = sc['rates']
        price_scenarios = list(map(float, st.text_input("أسعار (مفصولة بفواصل)", value="50,75,100").split(",")))
        opex_scenarios = list(map(float, st.text_input("OPEX scenarios (مفصولة بفواصل)", value="3000,5000,8000").split(",")))
        if st.button("تشغيل تحليل الحساسية"):
            df_sens = run_price_sensitivity(base_rates, price_scenarios, opex_scenarios, sc.get('econ', {}).get('capex', 0), discount_annual=sc.get('econ', {}).get('discount_annual', 0.10))
            st.dataframe(df_sens)
            log_event("INFO", "Sensitivity analysis run", scenario=sc_name)
    else:
        st.info("أضف سيناريو أو استورد بيانات لتشغيل تحليل الحساسية.")

# End of file

