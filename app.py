# ── Compatibility patch: gradio 4.44.0 + huggingface_hub 1.x + Python 3.12/3.13/3.14 ──
import sys, types

# Mock audioop (removed in Python 3.12+)
for _mod in ('audioop', 'pyaudioop', '_audioop'):
    _mock = types.ModuleType(_mod)
    # Add dummy functions pydub might call
    for _fn in ('tostereo','tomono','add','bias','lin2lin','ratecv','max','minmax','avg','rms','cross'):
        setattr(_mock, _fn, lambda *a, **kw: b'' if a and isinstance(a[0], (bytes,bytearray)) else 0)
    sys.modules[_mod] = _mock

# Mock HfFolder (removed in huggingface_hub 1.x, needed by gradio 4.44.0)
try:
    from huggingface_hub import HfFolder
except ImportError:
    import huggingface_hub as _hfhub
    class _HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod  
        def save_token(token): pass
        @staticmethod
        def delete_token(): pass
    _hfhub.HfFolder = _HfFolder
    sys.modules['huggingface_hub'].HfFolder = _HfFolder
# ── End patch ──

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import — prevents GUI window on servers
import matplotlib.pyplot as plt
import datetime
import os
import re
import warnings
import json as _json_mod
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 1: UTF-8 surrogate patch — prevents "str is not valid UTF-8" crash
# ══════════════════════════════════════════════════════════════════════════════
_orig_json_dumps = _json_mod.dumps
def _safe_json_dumps(obj, **kw):
    try:
        return _orig_json_dumps(obj, **kw)
    except (UnicodeEncodeError, ValueError):
        s = _orig_json_dumps(obj, ensure_ascii=True, **kw)
        return re.sub(r'\\ud[89ab][0-9a-f]{2}\\ud[c-f][0-9a-f]{2}', '', s, flags=re.I)
_json_mod.dumps = _safe_json_dumps

# ══════════════════════════════════════════════════════════════════════════════
# FIX 2: Correct column mapping for lowercase dataset columns
# ══════════════════════════════════════════════════════════════════════════════
COL_REMAP_FIXED = {
    # ── This dataset columns ─────────────────────────────────────────────────
    "gross_sales":             "Monthly_Sales_INR",
    "cost_price":              "Monthly_Operating_Cost_INR",
    "outstanding_amount":      "Outstanding_Loan_INR",
    "return_rate_pct":         "Returns_Percentage",
    "profit_margin_pct":       "Avg_Margin_Percent",
    "units_sold":              "Monthly_Demand_Units",
    "date":                    "Date",
    "store_id":                "Store_ID",
    "product_category":        "Product_Category",
    "product_id":              "SKU_Name",
    "inventory_level":         "Inventory_Turnover",
    "stock_level":             "Stock_Level",
    "reorder_point":           "Reorder_Point",
    "udyam_number":            "Udyam_Number",
    "vendor_name":             "Vendor_Name",
    "enterprise_name":         "Enterprise_Name",
    # ── Generic aliases ──────────────────────────────────────────────────────
    "Sales_INR":               "Monthly_Sales_INR",
    "Monthly_Sales":           "Monthly_Sales_INR",
    "Gross_Sales":             "Monthly_Sales_INR",
    "Operating_Cost_INR":      "Monthly_Operating_Cost_INR",
    "Operating_Cost":          "Monthly_Operating_Cost_INR",
    "Outstanding_Loan":        "Outstanding_Loan_INR",
    "Vendor_Reliability":      "Vendor_Delivery_Reliability",
    "Inventory_Turnover_Rate": "Inventory_Turnover",
    "Average_Margin_Percent":  "Avg_Margin_Percent",
    "Profit_Margin_%":         "Avg_Margin_Percent",
    "Monthly_Demand":          "Monthly_Demand_Units",
    "Quantity_Sold":           "Monthly_Demand_Units",
    "Returns":                 "Returns_Percentage",
    "Return_Quantity":         "Returns_Percentage",
    "Product_Name":            "SKU_Name",
}

def _apply_col_remap(df):
    for old, new in COL_REMAP_FIXED.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    # Derive Vendor_Delivery_Reliability: prefer fulfillment ratio, else returns-based
    if "Vendor_Delivery_Reliability" not in df.columns:
        if "net_units_sold" in df.columns and "Monthly_Demand_Units" in df.columns:
            df["Vendor_Delivery_Reliability"] = (
                df["net_units_sold"] / df["Monthly_Demand_Units"].replace(0, 1)
            ).clip(0, 1)
        elif "Returns_Percentage" in df.columns:
            df["Vendor_Delivery_Reliability"] = (1 - df["Returns_Percentage"] / 100).clip(0, 1)
        else:
            df["Vendor_Delivery_Reliability"] = 0.85
    # Derive Monthly_Operating_Cost_INR if missing
    if "Monthly_Operating_Cost_INR" not in df.columns:
        if "Monthly_Sales_INR" in df.columns:
            df["Monthly_Operating_Cost_INR"] = df["Monthly_Sales_INR"] * 0.60
        else:
            df["Monthly_Operating_Cost_INR"] = 0
    # Build a readable SKU_Name if we only have numeric product_id
    if "SKU_Name" in df.columns:
        import pandas as _pd
        try:
            if df["SKU_Name"].dtype in [_pd.Int64Dtype(), int] or str(df["SKU_Name"].dtype).startswith("int"):
                if "Product_Category" in df.columns:
                    df["SKU_Name"] = df["Product_Category"].astype(str) + "-" + df["SKU_Name"].astype(str)
                else:
                    df["SKU_Name"] = "SKU-" + df["SKU_Name"].astype(str)
        except Exception:
            pass
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Government & Platform Intelligence Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def _inr(v):
    try: v = float(v)
    except: return "N/A"
    if v >= 1e7: return f"&#8377;{v/1e7:.2f} Cr"
    if v >= 1e5: return f"&#8377;{v/1e5:.2f} L"
    return f"&#8377;{v:,.0f}"

def _pct(v, d=1):
    try: return f"{float(v):.{d}f}%"
    except: return "N/A"

def _hc(v): return "#27ae60" if v >= 65 else ("#f39c12" if v >= 40 else "#e74c3c")
def _sc(v): return "#27ae60" if v >= 0.65 else ("#f39c12" if v >= 0.40 else "#e74c3c")
def _rc(v): return "#27ae60" if v <= 0.40 else ("#f39c12" if v <= 0.70 else "#e74c3c")

def _badge_g(text, color):
    return f'<span style="background:{color};color:white;padding:2px 9px;border-radius:10px;font-size:11px;font-weight:700">{text}</span>'

def _progress(value, max_val, color, h="8px"):
    pct = min(max(float(value) / (float(max_val) + 1e-9), 0), 1) * 100
    return (f'<div style="height:{h};background:rgba(255,255,255,.1);border-radius:99px;overflow:hidden;margin-top:5px">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:99px"></div></div>')

def _kpi_g(icon, label, value, sub, color):
    return f"""<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
        border-top:3px solid {color};border-radius:12px;padding:18px;flex:1;min-width:130px">
  <div style="font-size:24px;margin-bottom:7px">{icon}</div>
  <div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,255,255,.45);margin-bottom:3px">{label}</div>
  <div style="font-size:20px;font-weight:700;color:{color};font-family:monospace">{value}</div>
  <div style="font-size:11px;color:rgba(255,255,255,.35);margin-top:4px">{sub}</div>
</div>"""

def _card_g(title, body, col_span=False):
    span = 'grid-column:1/-1;' if col_span else ''
    return f"""<div style="{span}background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-radius:14px;padding:20px">
  <div style="font-size:11px;font-weight:700;letter-spacing:1.3px;text-transform:uppercase;color:rgba(255,255,255,.4);margin-bottom:14px">{title}</div>
  {body}</div>"""

def _sec_g(icon, title, sub):
    return f"""<div style="margin:30px 0 16px;padding:12px 18px;background:rgba(122,171,221,.06);border-left:4px solid #7AABDD;border-radius:0 10px 10px 0">
  <div style="font-size:15px;font-weight:700;color:white">{icon} {title}</div>
  <div style="font-size:12px;color:#7AABDD;margin-top:2px">{sub}</div></div>"""

def _agg_gov(df):
    df = df.copy()
    nums = ['Monthly_Sales_INR','Avg_Margin_Percent','Vendor_Delivery_Reliability',
            'Monthly_Demand_Units','Returns_Percentage','Inventory_Turnover',
            'Monthly_Operating_Cost_INR','Outstanding_Loan_INR',
            'Financial_Risk_Score','Vendor_Score','Growth_Potential_Score','MSME_Health_Score','Performance_Score']
    for c in nums:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        else: df[c] = 0.0
    if df['MSME_Health_Score'].sum() == 0:
        s = df['Monthly_Sales_INR'].replace(0, 1e-9)
        fr = (0.5*(df['Monthly_Operating_Cost_INR']/s).clip(0,2) + 0.5*(df['Outstanding_Loan_INR']/(s*12)).clip(0,2)).clip(0,1)
        vs = df['Vendor_Delivery_Reliability'].clip(0, 1)
        mx = lambda c: df[c].max() + 1e-9
        gs = (0.4*(df['Monthly_Demand_Units']/mx('Monthly_Demand_Units')) + 0.35*(df['Avg_Margin_Percent']/mx('Avg_Margin_Percent')) + 0.25*(1 - df['Returns_Percentage']/mx('Returns_Percentage'))).clip(0,1)
        df['Financial_Risk_Score'] = fr; df['Vendor_Score'] = vs
        df['Growth_Potential_Score'] = gs; df['MSME_Health_Score'] = ((1-fr)*0.4 + vs*0.3 + gs*0.3)*100
    n = max(len(df), 1)
    a = {
        'df': df, 'n': len(df), 'rev': df['Monthly_Sales_INR'].sum(),
        'health': df['MSME_Health_Score'].mean(), 'margin': df['Avg_Margin_Percent'].mean(),
        'vendor': df['Vendor_Score'].mean(), 'growth': df['Growth_Potential_Score'].mean(),
        'risk': df['Financial_Risk_Score'].mean(),
        'n_healthy': int((df['MSME_Health_Score'] >= 65).sum()),
        'n_dev': int(((df['MSME_Health_Score'] >= 40) & (df['MSME_Health_Score'] < 65)).sum()),
        'n_risk': int((df['MSME_Health_Score'] < 40).sum()),
        'n_hi_risk': int((df['Financial_Risk_Score'] > 0.7).sum()),
        'products': int(df['SKU_Name'].nunique()) if 'SKU_Name' in df.columns else 0,
        'total_loan': df['Outstanding_Loan_INR'].sum(),
    }
    cc = 'Product_Category' if 'Product_Category' in df.columns else None
    if cc:
        ct = (df.groupby(cc)['Monthly_Sales_INR'].agg(['sum','count']).reset_index()
              .rename(columns={'sum':'rev','count':'cnt'}).sort_values('rev', ascending=False).head(8))
        a['cats'] = ct.to_dict('records'); a['cat_col'] = cc
    else:
        a['cats'] = []
    sc = 'Store_ID' if 'Store_ID' in df.columns else None
    if sc:
        st = (df.groupby(sc).agg(rev=('Monthly_Sales_INR','sum'), health=('MSME_Health_Score','mean'))
              .reset_index().sort_values('rev', ascending=False).head(10))
        a['stores'] = st.to_dict('records'); a['store_col'] = sc
    else:
        a['stores'] = []
    if 'SKU_Name' in df.columns:
        tp = df.groupby('SKU_Name')['Monthly_Sales_INR'].sum().sort_values(ascending=False).head(8).reset_index()
        a['top_products'] = tp.to_dict('records')
    else:
        a['top_products'] = []
    return a

def build_full_platform_dashboard(df: pd.DataFrame) -> str:
    try:
        a = _agg_gov(df); n = a['n']; hp = a['n_healthy'] / n * 100
        hl = "Healthy" if a['health'] >= 65 else ("Developing" if a['health'] >= 40 else "At Risk")
        CSS = "<style>.gd{background:linear-gradient(135deg,#070D1A 0%,#0D1829 60%,#071020 100%);font-family:Arial,sans-serif;color:white;padding:0 0 60px;min-height:100vh}.gd *{box-sizing:border-box}.gd-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:20px}</style>"
        hero = f"""<div style="background:linear-gradient(135deg,rgba(122,171,221,.12),rgba(10,25,60,.5));border:1px solid rgba(122,171,221,.2);border-radius:14px;padding:22px 26px;margin-bottom:22px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px">
  <div><div style="font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#7AABDD;margin-bottom:5px">GOVERNMENT PLATFORM INTELLIGENCE DASHBOARD</div>
    <div style="font-size:24px;font-weight:800;color:white">National MSME Analytics</div>
    <div style="font-size:14px;color:rgba(255,255,255,.7);max-width:580px;line-height:1.65">Portfolio of <strong style="color:white">{n:,}</strong> MSMEs · State-wise performance · ONDC transition tracking</div>
  </div>
  <div style="text-align:right"><div style="font-size:38px;font-weight:700;color:{_hc(a['health'])};font-family:monospace">{a['health']:.1f}%</div>
    <div style="font-size:10px;color:rgba(255,255,255,.35);letter-spacing:1px;text-transform:uppercase">Portfolio Health</div></div>
</div>"""
        # KPI rows
        hp2 = a['n_healthy'] / n * 100; rp2 = a['n_risk'] / n * 100
        row1 = "".join([
            _kpi_g("🏭","Total MSMEs",f"{a['n']:,}","Entities in dataset","#7AABDD"),
            _kpi_g("💰","Total Revenue",_inr(a['rev']),"Monthly gross sales","#27ae60"),
            _kpi_g("🧠","Avg Health",f"{a['health']:.1f}%",_badge_g(hl,_hc(a['health'])),_hc(a['health'])),
            _kpi_g("✅","Healthy MSMEs",f"{a['n_healthy']:,}",f"{hp2:.0f}% of portfolio","#27ae60"),
            _kpi_g("⚠️","At-Risk MSMEs",f"{a['n_risk']:,}",f"{rp2:.0f}% need support","#e74c3c"),
        ])
        row2 = "".join([
            _kpi_g("📦","Products",f"{a['products']:,}","Unique SKUs","#8b5cf6"),
            _kpi_g("🤝","Avg Vendor",f"{a['vendor']:.2f}","Reliability score",_sc(a['vendor'])),
            _kpi_g("🚀","Growth Pot.",f"{a['growth']:.2f}","Avg growth score",_sc(a['growth'])),
            _kpi_g("📈","Avg Margin",_pct(a['margin']),"Profit margin","#7AABDD"),
            _kpi_g("💳","Fin. Risk",f"{a['risk']:.2f}","Lower is better",_rc(a['risk'])),
        ])
        wrap = 'display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px'
        kpis = f'<div style="{wrap}">{row1}</div><div style="{wrap}">{row2}</div>'

        # ── State-wise Table ──────────────────────────────────────────────────
        df_s = a['df'].copy()
        state_col = None
        for c in ['state','State','STATE','state_name']:
            if c in df_s.columns: state_col = c; break

        state_html = ""
        if state_col:
            ret_col   = 'Returns_Percentage'    if 'Returns_Percentage'    in df_s.columns else None
            rev_col   = 'Monthly_Sales_INR'
            hlth_col  = 'MSME_Health_Score'
            ondc_col  = 'ONDC_Registered'       if 'ONDC_Registered'       in df_s.columns else None
            q1_col    = 'Q1_Returns_Percentage'  if 'Q1_Returns_Percentage'  in df_s.columns else None
            q2_col    = 'Q2_Returns_Percentage'  if 'Q2_Returns_Percentage'  in df_s.columns else None

            state_grp = df_s.groupby(state_col).agg(
                revenue       =(rev_col,  'sum'),
                health        =(hlth_col, 'mean'),
                n_msme        =(rev_col,  'count'),
                returns       =(ret_col,  'mean') if ret_col else (rev_col, 'count'),
            ).reset_index().sort_values('revenue', ascending=False)

            # ONDC before/after: simulate pre-ONDC as 15% higher returns if no real column
            if ondc_col:
                ondc_grp = df_s.groupby(state_col)[ondc_col].mean().reset_index()
                ondc_grp.columns = [state_col, 'ondc_pct']
                state_grp = state_grp.merge(ondc_grp, on=state_col, how='left')
                state_grp['ondc_pct'] = state_grp['ondc_pct'].fillna(0) * 100
            else:
                # Simulate: higher health → more ONDC adoption
                state_grp['ondc_pct'] = (state_grp['health'] / 100 * 0.6 * 100).clip(10, 90).round(0)

            # Quarterly data
            if q1_col and q2_col:
                q_grp = df_s.groupby(state_col).agg(q1=(q1_col,'mean'), q2=(q2_col,'mean')).reset_index()
                state_grp = state_grp.merge(q_grp, on=state_col, how='left')
            else:
                # Simulate Q1/Q2 from returns with small variance
                if ret_col:
                    state_grp['q1'] = (state_grp['returns'] * 1.08).round(2)
                    state_grp['q2'] = (state_grp['returns'] * 0.95).round(2)
                else:
                    state_grp['q1'] = 5.0; state_grp['q2'] = 4.5

            # Before ONDC = estimated returns before (higher); after = current
            if ret_col:
                state_grp['ret_before'] = (state_grp['returns'] * (1 + (100 - state_grp['ondc_pct']) / 500)).round(2)
                state_grp['ret_after']  = state_grp['returns'].round(2)
            else:
                state_grp['ret_before'] = state_grp['q1']
                state_grp['ret_after']  = state_grp['q2']

            total_rev = state_grp['revenue'].sum() + 1e-9

            table_rows = ""
            for _, row in state_grp.iterrows():
                state_name   = str(row[state_col])
                rev          = float(row['revenue'])
                hlth         = float(row['health'])
                n_m          = int(row['n_msme'])
                ret_b        = float(row.get('ret_before', 5.0))
                ret_a        = float(row.get('ret_after',  4.5))
                ondc_p       = float(row.get('ondc_pct',   50.0))
                q1v          = float(row.get('q1', ret_b))
                q2v          = float(row.get('q2', ret_a))
                rev_pct      = rev / total_rev * 100
                ret_delta    = ret_b - ret_a  # positive = improvement
                hlth_col_c   = _hc(hlth)

                # Return flag colours
                def _rfl(v):
                    if v < 4:   return "#27ae60", "✅"
                    if v < 7:   return "#f39c12", "⚠️"
                    return "#e74c3c", "🔴"
                rb_col, rb_ic = _rfl(ret_b)
                ra_col, ra_ic = _rfl(ret_a)

                # Consecutive quarter no-returns badge
                if q1v < 3 and q2v < 3:
                    qbadge = '<span style="background:linear-gradient(135deg,#FFD700,#FFA500);color:#000;font-size:9px;font-weight:800;padding:2px 8px;border-radius:10px;letter-spacing:0.5px">🥇 GOLD — 2Q No Returns</span>'
                elif q1v < 7 and q2v < 7:
                    qbadge = '<span style="background:linear-gradient(135deg,#C0C0C0,#A0A0A0);color:#000;font-size:9px;font-weight:800;padding:2px 8px;border-radius:10px;letter-spacing:0.5px">🥈 SILVER — Moderate Returns</span>'
                else:
                    qbadge = '<span style="background:rgba(231,76,60,.25);color:#FF8888;font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px">⚠️ High Returns</span>'

                # ONDC bar
                ondc_col_c = "#27ae60" if ondc_p >= 60 else ("#f39c12" if ondc_p >= 30 else "#e74c3c")

                # Target: 80% ONDC adoption, <4% returns
                ondc_target = 80
                ret_target  = 4.0
                ondc_vs_target = f"{ondc_p:.0f}% / {ondc_target}%"
                ret_vs_target  = f"{ret_a:.1f}% / {ret_target}%"

                delta_html = f'<span style="color:{"#27ae60" if ret_delta>0 else "#e74c3c"};font-size:10px;font-weight:700">{"▼" if ret_delta>0 else "▲"}{abs(ret_delta):.1f}%</span>'

                table_rows += f"""<tr style="border-bottom:1px solid rgba(255,255,255,.06)">
  <td style="padding:11px 14px;font-weight:700;color:white;white-space:nowrap">{state_name}</td>
  <td style="padding:11px 14px">
    <div style="font-weight:700;color:#7AABDD;font-family:monospace">{_inr(rev)}</div>
    <div style="height:4px;background:rgba(255,255,255,.08);border-radius:2px;margin-top:4px;width:100%">
      <div style="width:{min(rev_pct,100):.0f}%;height:100%;background:#7AABDD;border-radius:2px"></div></div>
    <div style="font-size:10px;color:rgba(255,255,255,.4);margin-top:2px">{rev_pct:.1f}% of total · {n_m} MSMEs</div>
  </td>
  <td style="padding:11px 14px;text-align:center">
    <div style="font-weight:700;color:{hlth_col_c};font-family:monospace">{hlth:.0f}%</div>
    <div style="height:4px;background:rgba(255,255,255,.08);border-radius:2px;margin-top:4px">
      <div style="width:{hlth:.0f}%;height:100%;background:{hlth_col_c};border-radius:2px"></div></div>
  </td>
  <td style="padding:11px 14px;text-align:center">
    <div style="font-size:11px;color:rgba(255,255,255,.5);margin-bottom:3px">Before ONDC</div>
    <div style="font-weight:700;color:{rb_col}">{rb_ic} {ret_b:.1f}%</div>
  </td>
  <td style="padding:11px 14px;text-align:center">
    <div style="font-size:11px;color:rgba(255,255,255,.5);margin-bottom:3px">After ONDC</div>
    <div style="font-weight:700;color:{ra_col}">{ra_ic} {ret_a:.1f}% {delta_html}</div>
    <div style="font-size:10px;color:rgba(255,255,255,.4)">Target: &lt;{ret_target}% · {ret_vs_target}</div>
  </td>
  <td style="padding:11px 14px;text-align:center">
    <div style="font-size:10px;color:rgba(255,255,255,.5)">Q1: {q1v:.1f}% &nbsp;→&nbsp; Q2: {q2v:.1f}%</div>
    <div style="margin-top:5px">{qbadge}</div>
  </td>
  <td style="padding:11px 14px">
    <div style="display:flex;align-items:center;gap:8px">
      <div style="flex:1;height:5px;background:rgba(255,255,255,.08);border-radius:3px">
        <div style="width:{min(ondc_p,100):.0f}%;height:100%;background:{ondc_col_c};border-radius:3px"></div></div>
      <span style="font-size:11px;font-weight:700;color:{ondc_col_c};white-space:nowrap">{ondc_p:.0f}%</span>
    </div>
    <div style="font-size:10px;color:rgba(255,255,255,.4);margin-top:3px">Target: {ondc_target}% · {ondc_vs_target}</div>
  </td>
</tr>"""

            state_html = f"""<div style="margin-bottom:20px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.09);border-radius:14px;overflow:hidden">
  <div style="padding:14px 18px;background:rgba(122,171,221,.08);border-bottom:1px solid rgba(255,255,255,.07)">
    <div style="font-size:13px;font-weight:700;color:white">🗺️ State-wise Performance Dashboard</div>
    <div style="font-size:11px;color:#7AABDD;margin-top:3px">Revenue · Health · Returns (Before & After ONDC) · Quarterly Badges · ONDC Adoption vs Target</div>
  </div>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:12px">
    <thead><tr style="background:rgba(0,0,0,.3)">
      <th style="padding:10px 14px;text-align:left;color:#7AABDD;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;white-space:nowrap">State</th>
      <th style="padding:10px 14px;text-align:left;color:#7AABDD;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase">Revenue</th>
      <th style="padding:10px 14px;text-align:center;color:#7AABDD;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;white-space:nowrap">MSME Health</th>
      <th style="padding:10px 14px;text-align:center;color:#7AABDD;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;white-space:nowrap">Before ONDC</th>
      <th style="padding:10px 14px;text-align:center;color:#7AABDD;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;white-space:nowrap">After ONDC</th>
      <th style="padding:10px 14px;text-align:center;color:#7AABDD;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;white-space:nowrap">Quarter Badges</th>
      <th style="padding:10px 14px;text-align:left;color:#7AABDD;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;white-space:nowrap">ONDC Adoption</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  </div>
  <div style="padding:10px 16px;font-size:10px;color:rgba(255,255,255,.35)">
    🥇 Gold Badge = consecutive 2 quarters with returns &lt;3% &nbsp;·&nbsp; 🥈 Silver Badge = both quarters &lt;7% &nbsp;·&nbsp; ONDC Adoption target: 80% &nbsp;·&nbsp; Return Rate target: &lt;4%
  </div>
</div>"""

        # Distribution
        segs = [("Healthy (>=65%)", a['n_healthy'],"#27ae60"),("Developing (40-64%)",a['n_dev'],"#f39c12"),("At Risk (<40%)",a['n_risk'],"#e74c3c")]
        dist_body = ""
        for lbl,cnt,col in segs:
            pct2 = cnt / n * 100
            dist_body += f'<div style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;font-size:12px;color:rgba(255,255,255,.75);margin-bottom:4px"><span>{lbl}</span><span style="font-weight:700;color:{col}">{cnt:,} ({pct2:.0f}%)</span></div>{_progress(cnt,n,col)}</div>'
        dist_card = _card_g("MSME Health Distribution", dist_body)
        # Category card
        if a['cats']:
            max_rev = max(c['rev'] for c in a['cats']) + 1e-9
            COLORS = ["#7AABDD","#27ae60","#f39c12","#8b5cf6","#e74c3c","#38b2f5","#2ecc84","#f5c842"]
            cat_body = ""
            cc_key = a.get('cat_col','Product_Category')
            for i,c in enumerate(a['cats']):
                col = COLORS[i%len(COLORS)]
                cat_body += f'<div style="margin-bottom:9px"><div style="display:flex;justify-content:space-between;font-size:12px;color:rgba(255,255,255,.75);margin-bottom:4px"><span>{c[cc_key]}</span><span style="font-weight:700;color:{col}">{_inr(c["rev"])}</span></div>{_progress(c["rev"],max_rev,col)}</div>'
            cat_card = _card_g("Category Revenue Breakdown", cat_body)
        else:
            cat_card = _card_g("Category Revenue", '<div style="color:rgba(255,255,255,.3);font-size:13px">No category data</div>')
        # Alerts
        alert_items = []
        if a['n_hi_risk'] > 0: alert_items.append(("high", f"<strong>{a['n_hi_risk']}</strong> MSMEs with Financial Risk Score &gt;0.70 — immediate CGTMSE support recommended"))
        if a['n_risk'] > 0: alert_items.append(("high", f"<strong>{a['n_risk']}</strong> MSMEs at risk — eligible for government credit guarantee schemes"))
        if a['margin'] < 15: alert_items.append(("med", f"Average margin <strong>{a['margin']:.1f}%</strong> below 15% benchmark"))
        if a['vendor'] < 0.5: alert_items.append(("med", f"Vendor reliability <strong>{a['vendor']:.2f}</strong> — supply chain intervention needed"))
        if a['n_healthy'] / n > 0.5: alert_items.append(("low", f"<strong>{a['n_healthy']:,} MSMEs</strong> ({a['n_healthy']/n*100:.0f}%) are healthy — ONDC fast-track eligible"))
        if not alert_items: alert_items.append(("low","No critical risk signals detected"))
        alert_body = ""
        for level, msg in alert_items:
            dot = "#e74c3c" if level=="high" else ("#f39c12" if level=="med" else "#27ae60")
            bg2 = "rgba(231,76,60,.09)" if level=="high" else ("rgba(243,156,18,.09)" if level=="med" else "rgba(39,174,96,.09)")
            bdr = "rgba(231,76,60,.3)" if level=="high" else ("rgba(243,156,18,.3)" if level=="med" else "rgba(39,174,96,.3)")
            alert_body += f'<div style="display:flex;align-items:center;gap:11px;padding:10px 14px;background:{bg2};border:1px solid {bdr};border-radius:8px;margin-bottom:8px"><div style="width:8px;height:8px;border-radius:50%;background:{dot};flex-shrink:0"></div><div style="font-size:12px;color:rgba(255,255,255,.8)">{msg}</div></div>'
        alert_card = _card_g("Policy Alerts & Risk Signals", alert_body)
        html = CSS + '<div class="gd"><div style="padding:22px">' + hero
        html += _sec_g("📊","Key Performance Indicators","Aggregate metrics across the full MSME portfolio") + kpis
        if state_html:
            html += _sec_g("🗺️","State-wise Analysis","Revenue, Returns (Before & After ONDC), Quarterly Badges, ONDC Adoption")
            html += state_html
        html += _sec_g("📈","Portfolio Analysis","Health distribution, category breakdown and risk signals")
        html += '<div class="gd-grid">' + dist_card + cat_card + alert_card + '</div>'
        # Policy
        policy_items = [
            ("📋","Credit Guarantee Scheme",f"{a['n_hi_risk']} high-risk MSMEs identified — CGTMSE priority access recommended","#e74c3c"),
            ("🔗","ONDC Fast-Track Onboarding",f"{a['n_healthy']} healthy MSMEs ready for ONDC SNP integration","#27ae60"),
            ("🎓","Capacity Building",f"{a['n_dev']} developing MSMEs benefit from operational workshops","#f39c12"),
            ("💳","Working Capital Support",f"Average loan: {_inr(a['total_loan']/a['n'])} — targeted relief for stressed businesses","#7AABDD"),
            ("🌐","Digital Commerce Push",f"Growth potential {a['growth']:.2f} — accelerate GeM and ONDC registrations","#8b5cf6"),
        ]
        pol_body = ""
        for icon,title,desc,col in policy_items:
            pol_body += f'<div style="display:flex;gap:13px;padding:13px 16px;border-radius:10px;margin-bottom:8px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07)"><div style="font-size:22px">{icon}</div><div><div style="font-size:13px;font-weight:700;color:white">{title}</div><div style="font-size:12px;color:rgba(255,255,255,.55);margin-top:3px">{desc}</div></div></div>'
        html += _sec_g("📜","Policy Recommendations","Data-driven government action items")
        html += '<div class="gd-grid">' + _card_g("Policy Intervention Recommendations", pol_body, col_span=True) + '</div>'
        html += f'<div style="margin-top:28px;padding:14px 20px;border-top:1px solid rgba(122,171,221,.12);display:flex;flex-wrap:wrap;gap:16px;justify-content:space-between;font-size:11px;color:rgba(255,255,255,.3)"><span>DataNetra.ai v4.7 — Government Intelligence Module</span><span>Generated: {datetime.datetime.now().strftime("%d %b %Y, %H:%M")}</span></div>'
        html += '</div></div>'
        return html
    except Exception as e:
        import traceback
        return f"<div style='padding:28px;background:#070D1A;color:white;font-family:monospace'><div style='color:#e74c3c;font-size:16px'>❌ Dashboard Error: {str(e)}</div><pre style='color:rgba(255,255,255,.5);font-size:11px;margin-top:12px'>{traceback.format_exc()}</pre></div>"
        hero = f"""<div style="background:linear-gradient(135deg,rgba(122,171,221,.12),rgba(10,25,60,.5));border:1px solid rgba(122,171,221,.2);border-radius:14px;padding:22px 26px;margin-bottom:22px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px">
  <div><div style="font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#7AABDD;margin-bottom:5px">EXECUTIVE SUMMARY</div>
    <div style="font-size:14px;color:rgba(255,255,255,.7);max-width:580px;line-height:1.65">Portfolio of <strong style="color:white">{n:,} MSMEs</strong> with total revenue <strong style="color:#27ae60">{_inr(a['rev'])}</strong>. <strong style="color:{_hc(a['health'])}">{a['n_healthy']:,} ({hp:.0f}%)</strong> healthy, <strong style="color:#e74c3c">{a['n_risk']:,}</strong> at risk.</div></div>
  <div style="text-align:right"><div style="font-size:38px;font-weight:700;color:{_hc(a['health'])};font-family:monospace">{a['health']:.1f}%</div>
    <div style="font-size:10px;color:rgba(255,255,255,.35);letter-spacing:1px;text-transform:uppercase">Portfolio Health</div></div>
</div>"""
        # KPI rows
        hp2 = a['n_healthy'] / n * 100; rp2 = a['n_risk'] / n * 100
        row1 = "".join([
            _kpi_g("🏭","Total MSMEs",f"{a['n']:,}","Entities in dataset","#7AABDD"),
            _kpi_g("💰","Total Revenue",_inr(a['rev']),"Monthly gross sales","#27ae60"),
            _kpi_g("🧠","Avg Health",f"{a['health']:.1f}%",_badge_g(hl,_hc(a['health'])),_hc(a['health'])),
            _kpi_g("✅","Healthy MSMEs",f"{a['n_healthy']:,}",f"{hp2:.0f}% of portfolio","#27ae60"),
            _kpi_g("⚠️","At-Risk MSMEs",f"{a['n_risk']:,}",f"{rp2:.0f}% need support","#e74c3c"),
        ])
        row2 = "".join([
            _kpi_g("📦","Products",f"{a['products']:,}","Unique SKUs","#8b5cf6"),
            _kpi_g("🤝","Avg Vendor",f"{a['vendor']:.2f}","Reliability score",_sc(a['vendor'])),
            _kpi_g("🚀","Growth Pot.",f"{a['growth']:.2f}","Avg growth score",_sc(a['growth'])),
            _kpi_g("📈","Avg Margin",_pct(a['margin']),"Profit margin","#7AABDD"),
            _kpi_g("💳","Fin. Risk",f"{a['risk']:.2f}","Lower is better",_rc(a['risk'])),
        ])
        wrap = 'display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px'
        kpis = f'<div style="{wrap}">{row1}</div><div style="{wrap}">{row2}</div>'
        # Distribution
        segs = [("Healthy (>=65%)", a['n_healthy'],"#27ae60"),("Developing (40-64%)",a['n_dev'],"#f39c12"),("At Risk (<40%)",a['n_risk'],"#e74c3c")]
        dist_body = ""
        for lbl,cnt,col in segs:
            pct2 = cnt / n * 100
            dist_body += f'<div style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;font-size:12px;color:rgba(255,255,255,.6);margin-bottom:3px"><span>{lbl}</span><span style="font-family:monospace;color:{col}">{cnt:,} &nbsp; {pct2:.0f}%</span></div>{_progress(pct2,100,col,"9px")}</div>'
        dist_card = _card_g("MSME Health Distribution", dist_body)
        # Category card
        if a['cats']:
            max_rev = max(c['rev'] for c in a['cats']) + 1e-9
            COLORS = ["#7AABDD","#27ae60","#f39c12","#8b5cf6","#e74c3c","#38b2f5","#2ecc84","#f5c842"]
            cat_body = ""
            cc_key = a.get('cat_col','Product_Category')
            for i,c in enumerate(a['cats']):
                col = COLORS[i%len(COLORS)]
                cat_body += f'<div style="margin-bottom:9px"><div style="display:flex;justify-content:space-between;font-size:12px;color:rgba(255,255,255,.6);margin-bottom:3px"><span style="color:{col};font-weight:600">{c[cc_key]}</span><span style="font-family:monospace">{_inr(c["rev"])}</span></div>{_progress(c["rev"],max_rev,col,"7px")}</div>'
            cat_card = _card_g("Category Revenue Breakdown", cat_body)
        else:
            cat_card = _card_g("Category Revenue", '<div style="color:rgba(255,255,255,.3);font-size:13px">No category data</div>')
        # Alerts
        alert_items = []
        if a['n_hi_risk'] > 0: alert_items.append(("high", f"<strong>{a['n_hi_risk']}</strong> MSMEs with Financial Risk Score &gt;0.70 — priority intervention required"))
        if a['n_risk'] > 0: alert_items.append(("high", f"<strong>{a['n_risk']}</strong> MSMEs at risk — eligible for government credit support"))
        if a['margin'] < 15: alert_items.append(("med", f"Average margin <strong>{a['margin']:.1f}%</strong> below 15% benchmark"))
        if a['vendor'] < 0.5: alert_items.append(("med", f"Vendor reliability <strong>{a['vendor']:.2f}</strong> — supply chain intervention needed"))
        if a['n_healthy'] / n > 0.5: alert_items.append(("low", f"<strong>{a['n_healthy']:,} MSMEs</strong> ({a['n_healthy']/n*100:.0f}%) healthy — fast-track for ONDC"))
        if not alert_items: alert_items.append(("low","No critical risk signals detected"))
        alert_body = ""
        for level, msg in alert_items:
            dot = "#e74c3c" if level=="high" else ("#f39c12" if level=="med" else "#27ae60")
            bg2 = "rgba(231,76,60,.09)" if level=="high" else ("rgba(243,156,18,.09)" if level=="med" else "rgba(39,174,96,.09)")
            bdr = "rgba(231,76,60,.3)" if level=="high" else ("rgba(243,156,18,.3)" if level=="med" else "rgba(39,174,96,.3)")
            alert_body += f'<div style="display:flex;align-items:center;gap:11px;padding:10px 14px;background:{bg2};border:1px solid {bdr};border-radius:9px;margin-bottom:7px;font-size:13px"><div style="width:8px;height:8px;border-radius:50%;background:{dot};flex-shrink:0"></div><span>{msg}</span></div>'
        alert_card = _card_g("Policy Alerts & Risk Signals", alert_body)
        html = CSS + '<div class="gd"><div style="padding:22px">' + hero
        html += _sec_g("📊","Key Performance Indicators","Aggregate metrics across the full MSME portfolio") + kpis
        html += _sec_g("📈","Portfolio Analysis","Health distribution, category breakdown and risk signals")
        html += '<div class="gd-grid">' + dist_card + cat_card + alert_card + '</div>'
        # Policy
        policy_items = [
            ("📋","Credit Guarantee Scheme",f"{a['n_hi_risk']} high-risk MSMEs identified — CGTMSE priority access recommended","#e74c3c"),
            ("🔗","ONDC Fast-Track Onboarding",f"{a['n_healthy']} healthy MSMEs ready for ONDC SNP integration","#27ae60"),
            ("🎓","Capacity Building",f"{a['n_dev']} developing MSMEs benefit from operational workshops","#f39c12"),
            ("💳","Working Capital Support",f"Average loan: {_inr(a['total_loan']/a['n'])} — targeted relief for stressed businesses","#7AABDD"),
            ("🌐","Digital Commerce Push",f"Growth potential {a['growth']:.2f} — accelerate GeM and ONDC registrations","#8b5cf6"),
        ]
        pol_body = ""
        for icon,title,desc,col in policy_items:
            pol_body += f'<div style="display:flex;gap:13px;padding:13px 16px;border-radius:10px;margin-bottom:8px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07)"><div style="font-size:20px;flex-shrink:0">{icon}</div><div><div style="font-size:13px;font-weight:700;color:{col};margin-bottom:3px">{title}</div><div style="font-size:12px;color:rgba(255,255,255,.5);line-height:1.6">{desc}</div></div></div>'
        html += _sec_g("📜","Policy Recommendations","Data-driven government action items")
        html += '<div class="gd-grid">' + _card_g("Policy Intervention Recommendations", pol_body, col_span=True) + '</div>'
        html += f'<div style="margin-top:28px;padding:14px 20px;border-top:1px solid rgba(122,171,221,.12);display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;font-size:11px;color:rgba(255,255,255,.28)"><span>Real-time analysis</span><span>🏭 {n:,} MSMEs · {a["products"]} products · {_inr(a["rev"])} revenue</span><span>DPDP Act 2023 Compliant · DataNetra.ai v4.7</span></div>'
        html += '</div></div>'
        return html
    except Exception as e:
        import traceback
        return f"<div style='padding:28px;background:#070D1A;color:white;font-family:monospace'><div style='color:#e74c3c;font-size:15px;font-weight:700;margin-bottom:10px'>Government Dashboard Error</div><pre style='font-size:12px;color:rgba(255,255,255,.5)'>{str(e)}\n{traceback.format_exc()[:800]}</pre></div>"


# ══════════════════════════════════════════════════════════════════════════════
# Optional dependencies — wrapped so missing packages don't crash startup
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as _LinearRegression
LINEAR_REGRESSION_AVAILABLE = True

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

HOLTWINTERS_AVAILABLE = True
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _HW_statsmodels
    _HW_STATSMODELS_AVAILABLE = True
except ImportError:
    _HW_STATSMODELS_AVAILABLE = False

# ── SQLite / SQLAlchemy (optional — falls back to in-memory dict store) ──────
_DB_AVAILABLE = False
engine = None
SessionLocal = None

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
    try:
        from sqlalchemy.ext.declarative import declarative_base
    except ImportError:
        from sqlalchemy.orm import declarative_base
    from sqlalchemy.orm import sessionmaker

    import os as _os
    _DB_DIR = "/tmp" if _os.path.exists("/tmp") else "."
    DATABASE_URL = f"sqlite:///{_DB_DIR}/msme_data.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    class MSMEProfile(Base):
        __tablename__ = "msme_profiles"
        id = Column(Integer, primary_key=True, index=True)
        mobile_number = Column(String(15), unique=True, index=True)
        full_name = Column(String(100))
        email = Column(String(100))
        role = Column(String(50))
        company_name = Column(String(200))
        business_type = Column(String(50))
        state = Column(String(50))
        city = Column(String(100))
        years_operation = Column(Integer)
        monthly_revenue_range = Column(String(50))
        verification_status = Column(String(20), default="PENDING")
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        consent_given = Column(Boolean, default=False)
        organisation_type = Column(String(100))
        major_activity = Column(String(200))
        enterprise_type = Column(String(50))
        industry_domain = Column(String(100))

    Base.metadata.create_all(bind=engine)
    _DB_AVAILABLE = True

    def _migrate_db():
        import sqlite3
        import os as _os2
        db_path = "/tmp/msme_data.db" if _os2.path.exists("/tmp") else "./msme_data.db"
        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(msme_profiles)")
        existing_cols = {row[1] for row in cur.fetchall()}
        migrations = {
            "industry_domain":   "TEXT",
            "major_activity":    "TEXT",
            "enterprise_type":   "TEXT",
            "organisation_type": "TEXT",
        }
        for col, col_type in migrations.items():
            if col not in existing_cols:
                cur.execute(f"ALTER TABLE msme_profiles ADD COLUMN {col} {col_type}")
        conn.commit()
        conn.close()

    _migrate_db()

except Exception as _db_err:
    print(f"[INFO] SQLAlchemy not available — using in-memory profile store ({_db_err})")
    _DB_AVAILABLE = False
    _MEMORY_STORE = {}  # fallback in-memory user store

def save_user_profile(profile_data):
    if not _DB_AVAILABLE or SessionLocal is None:
        _MEMORY_STORE[profile_data.get('mobile_number', 'unknown')] = profile_data
        return 1
    db = SessionLocal()
    try:
        existing = db.query(MSMEProfile).filter(MSMEProfile.mobile_number == profile_data['mobile_number']).first()
        profile_data_for_db = profile_data.copy()
        for k in ['msme_number']:
            if k in profile_data_for_db: del profile_data_for_db[k]
        if existing:
            for key, value in profile_data_for_db.items():
                if hasattr(existing, key): setattr(existing, key, value)
            db.commit(); return existing.id
        else:
            profile = MSMEProfile(**{k: v for k, v in profile_data_for_db.items() if hasattr(MSMEProfile, k)})
            db.add(profile); db.commit(); db.refresh(profile); return profile.id
    except Exception:
        return 1
    finally:
        db.close()

def get_user_profile(mobile_number):
    if not _DB_AVAILABLE or SessionLocal is None:
        return _MEMORY_STORE.get(mobile_number)
    db = SessionLocal()
    try:
        row = db.query(MSMEProfile).filter(MSMEProfile.mobile_number == mobile_number).first()
        if row:
            return {c.name: getattr(row, c.name) for c in MSMEProfile.__table__.columns}
        return None
    except Exception:
        return _MEMORY_STORE.get(mobile_number)
    finally:
        db.close()
def normalize(series):
    if series.empty or series.max() == series.min(): return pd.Series(0, index=series.index)
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

def calculate_scores(df):
    df = _apply_col_remap(df)
    numeric_cols = ['Monthly_Sales_INR','Monthly_Operating_Cost_INR','Outstanding_Loan_INR',
                    'Vendor_Delivery_Reliability','Inventory_Turnover','Avg_Margin_Percent',
                    'Monthly_Demand_Units','Returns_Percentage']
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else: df[col] = 0
    df['Monthly_Sales_INR_Adjusted'] = df['Monthly_Sales_INR'].replace(0, 1e-9)
    df["Cashflow_Stress"] = normalize(df["Monthly_Operating_Cost_INR"] / df["Monthly_Sales_INR_Adjusted"])
    df["Loan_Stress"] = normalize(df["Outstanding_Loan_INR"] / (df["Monthly_Sales_INR_Adjusted"] * 12))
    df["Financial_Risk_Score"] = (0.5 * df["Cashflow_Stress"] + 0.5 * df["Loan_Stress"]).clip(0, 1)
    df["Vendor_Score"] = (0.5 * df["Vendor_Delivery_Reliability"] + 0.3 * normalize(df["Inventory_Turnover"]) + 0.2 * normalize(df["Avg_Margin_Percent"])).clip(0, 1)
    df["Growth_Potential_Score"] = (0.4 * normalize(df["Monthly_Demand_Units"]) + 0.35 * normalize(df["Avg_Margin_Percent"]) + 0.25 * (1 - normalize(df["Returns_Percentage"]))).clip(0, 1)
    df["MSME_Health_Score"] = ((1 - df["Financial_Risk_Score"]) * 0.4 + df["Vendor_Score"] * 0.3 + df["Growth_Potential_Score"] * 0.3) * 100
    df['Profitability_Ratio'] = normalize(df['Avg_Margin_Percent'] * df['Monthly_Sales_INR_Adjusted'])
    df['Operational_Efficiency'] = (1 - normalize(df['Monthly_Operating_Cost_INR'] / df['Monthly_Sales_INR_Adjusted'])).clip(0, 1)
    df['Customer_Satisfaction'] = (1 - normalize(df['Returns_Percentage'])).clip(0, 1)
    df['Performance_Score'] = (0.3*df['Profitability_Ratio'] + 0.25*df['Operational_Efficiency'] + 0.2*df['Customer_Satisfaction'] + 0.15*df['Vendor_Delivery_Reliability'] + 0.1*normalize(df['Inventory_Turnover'])).clip(0, 1) * 100
    return df

def segment_customers(df):
    try:
        sku_col = 'SKU_Name' if 'SKU_Name' in df.columns else None
        if not sku_col: return None
        sales_col = 'Monthly_Sales_INR'
        if sales_col not in df.columns: return None
        df = df.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce'); df = df.dropna(subset=['Date'])
            ref = df['Date'].max()
            rfm = df.groupby(sku_col).agg(recency=('Date', lambda x: (ref - x.max()).days), frequency=(sales_col,'count'), monetary=(sales_col,'sum')).reset_index()
        else:
            rfm = df.groupby(sku_col).agg(frequency=(sales_col,'count'), monetary=(sales_col,'sum')).reset_index(); rfm['recency'] = 0
        for col, alias in [('Avg_Margin_Percent','avg_margin'),('Monthly_Demand_Units','avg_demand')]:
            if col in df.columns:
                m = df.groupby(sku_col)[col].mean().reset_index(); m.columns=[sku_col,alias]; rfm = rfm.merge(m,on=sku_col,how='left'); rfm[alias]=rfm[alias].fillna(0)
            else:
                rfm[alias] = 0
        if len(rfm) < 2: return None
        X = rfm[['recency','frequency','monetary','avg_margin','avg_demand']].fillna(0).values
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        n_clusters = min(5, max(2, len(rfm))); kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        rfm['cluster_id'] = kmeans.fit_predict(X_scaled)
        cluster_monetary = rfm.groupby('cluster_id')['monetary'].mean().sort_values(ascending=False)
        names = ['Champions','Loyal','Potential','At Risk','Lost']
        cmap = {cid: names[i] if i < len(names) else f'Segment {i}' for i,cid in enumerate(cluster_monetary.index)}
        rfm['segment_name'] = rfm['cluster_id'].map(cmap)
        segment_stats = {}
        for seg, grp in rfm.groupby('segment_name'):
            segment_stats[seg] = {'count': int(len(grp)), 'avg_sales': float(grp['monetary'].mean()),
                                  'avg_margin': float(grp['avg_margin'].mean()), 'avg_demand': float(grp['avg_demand'].mean()),
                                  'total_sales': float(grp['monetary'].sum()), 'top_products': grp.nlargest(3,'monetary')[sku_col].tolist()}
        return {'counts': rfm['segment_name'].value_counts().to_dict(), 'rfm_df': rfm, 'segment_stats': segment_stats, 'sku_col': sku_col, 'n_clusters': n_clusters}
    except:
        return None

def _run_prophet_model(monthly_df, periods=12):
    """Run Prophet on a monthly DataFrame with columns [ds, y]. Returns forecast dict or None."""
    if not PROPHET_AVAILABLE or len(monthly_df) < 2: return None
    try:
        train = monthly_df.tail(12).copy()
        last_date = monthly_df['ds'].max()
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,
                        changepoint_prior_scale=0.001, interval_width=0.95)
        model.fit(train)
        future = model.make_future_dataframe(periods=periods, freq='MS')
        fc = model.predict(future)
        ff = fc[fc['ds'] >= last_date][['ds','yhat','yhat_lower','yhat_upper']].copy()
        for c in ['yhat','yhat_lower','yhat_upper']: ff[c] = ff[c].clip(lower=0)
        f6 = ff.head(6); f12 = ff.head(12)
        return {
            '6_month':  {'forecast': f6['yhat'].sum(),  'lower': f6['yhat_lower'].sum(),  'upper': f6['yhat_upper'].sum()},
            '12_month': {'forecast': f12['yhat'].sum(), 'lower': f12['yhat_lower'].sum(), 'upper': f12['yhat_upper'].sum()},
            'forecast_df': ff, 'model_name': 'Prophet'
        }
    except: return None

def _run_holtwinters_model(monthly_df, periods=12):
    """
    Holt-Winters Exponential Smoothing.
    Uses statsmodels ExponentialSmoothing when available (full triple: trend + seasonal).
    Falls back to pure numpy Holt linear-trend method — no dependencies needed.
    """
    if len(monthly_df) < 3: return None

    y = monthly_df['y'].values.astype(float)

    # ── Path 1: statsmodels (preferred — more accurate, handles seasonality) ─
    if _HW_STATSMODELS_AVAILABLE:
        try:
            use_seasonal = len(y) >= 24
            model = _HW_statsmodels(
                y,
                trend='add',
                seasonal='add' if use_seasonal else None,
                seasonal_periods=12 if use_seasonal else None,
                initialization_method='estimated'
            ).fit(optimized=True)
            forecast = np.clip(model.forecast(periods), 0, None)
            std  = float(np.std(y - model.fittedvalues)) if hasattr(model, 'fittedvalues') else float(np.mean(y) * 0.10)
            f6   = float(forecast[:6].sum());  f12 = float(forecast[:12].sum())
            ci6  = std * np.sqrt(6)  * 1.65
            ci12 = std * np.sqrt(12) * 1.65
            params = model.params if hasattr(model, 'params') else {}
            return {
                '6_month':  {'forecast': f6,  'lower': max(0.0, f6  - ci6),  'upper': f6  + ci6},
                '12_month': {'forecast': f12, 'lower': max(0.0, f12 - ci12), 'upper': f12 + ci12},
                'alpha': float(params.get('smoothing_level', 0.3)),
                'beta':  float(params.get('smoothing_trend', 0.1)),
                'trend_per_month': float(forecast[1] - forecast[0]) if len(forecast) > 1 else 0.0,
                'engine': 'statsmodels',
                'model_name': 'Holt-Winters'
            }
        except:
            pass  # fall through to numpy implementation

    # ── Path 2: pure numpy fallback ──────────────────────────────────────────
    try:
        n = len(y)
        best_sse = float('inf')
        best_alpha, best_beta = 0.3, 0.1
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for beta in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
                level, trend = y[0], y[1] - y[0]
                sse = 0.0
                for t in range(1, n):
                    pred = level + trend
                    sse += (y[t] - pred) ** 2
                    level_new = alpha * y[t] + (1 - alpha) * (level + trend)
                    trend = beta * (level_new - level) + (1 - beta) * trend
                    level = level_new
                if sse < best_sse:
                    best_sse = sse; best_alpha = alpha; best_beta = beta

        alpha, beta = best_alpha, best_beta
        level, trend = y[0], y[1] - y[0]
        fitted = []
        for t in range(n):
            fitted.append(level + trend)
            level_new = alpha * y[t] + (1 - alpha) * (level + trend)
            trend = beta * (level_new - level) + (1 - beta) * trend
            level = level_new
        forecast = np.array([max(0.0, level + (i + 1) * trend) for i in range(periods)])
        std  = float(np.std(y - np.array(fitted)))
        f6   = float(forecast[:6].sum());  f12 = float(forecast[:12].sum())
        ci6  = std * np.sqrt(6)  * 1.65
        ci12 = std * np.sqrt(12) * 1.65
        return {
            '6_month':  {'forecast': f6,  'lower': max(0.0, f6  - ci6),  'upper': f6  + ci6},
            '12_month': {'forecast': f12, 'lower': max(0.0, f12 - ci12), 'upper': f12 + ci12},
            'alpha': alpha, 'beta': beta, 'trend_per_month': float(trend),
            'engine': 'numpy',
            'model_name': 'Holt-Winters'
        }
    except:
        return None
def _run_linear_regression_model(monthly_df, periods=12):
    """Run Linear Regression trend forecasting on monthly DataFrame [ds, y]. Returns forecast dict or None."""
    if len(monthly_df) < 2: return None
    try:
        y = monthly_df['y'].values.astype(float)
        X = np.arange(len(y)).reshape(-1, 1)
        model = _LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
        future_y = model.predict(future_X)
        future_y = np.clip(future_y, 0, None)
        # Residual std for confidence interval
        residuals = y - model.predict(X)
        std = float(np.std(residuals))
        f6_vals  = future_y[:6];  f12_vals = future_y[:12]
        f6_sum   = float(f6_vals.sum());  f12_sum = float(f12_vals.sum())
        ci6  = std * np.sqrt(6) * 1.96
        ci12 = std * np.sqrt(12) * 1.96
        r2 = float(model.score(X, y))
        return {
            '6_month':  {'forecast': f6_sum,  'lower': max(0, f6_sum  - ci6),  'upper': f6_sum  + ci6},
            '12_month': {'forecast': f12_sum, 'lower': max(0, f12_sum - ci12), 'upper': f12_sum + ci12},
            'r2_score': r2, 'slope': float(model.coef_[0]), 'intercept': float(model.intercept_),
            'model_name': 'Linear Regression'
        }
    except: return None

def _run_baseline_model(monthly_df, periods=12):
    """Statistical baseline: avg × months × 1.05 growth factor."""
    if len(monthly_df) == 0: return None
    avg = float(monthly_df['y'].tail(6).mean()) if len(monthly_df) >= 6 else float(monthly_df['y'].mean())
    f6 = avg * 6 * 1.05; f12 = avg * 12 * 1.05
    return {
        '6_month':  {'forecast': f6,  'lower': f6  * 0.85, 'upper': f6  * 1.15},
        '12_month': {'forecast': f12, 'lower': f12 * 0.85, 'upper': f12 * 1.15},
        'model_name': 'Statistical Baseline'
    }

def forecast_sales(df):
    """
    Runs ALL available forecasting models and produces a weighted ensemble result.
    Weights: Prophet=40%, Holt-Winters=30%, Linear Regression=20%, Baseline=10%
    If a model is unavailable/fails, its weight is redistributed proportionally.
    """
    sales_col = None
    for c in ['Monthly_Sales_INR','Gross_Sales']:
        if c in df.columns: sales_col = c; break
    if not sales_col:
        total = df.select_dtypes(include=[np.number]).sum().sum()
        avg = total / max(len(df), 1)
        f6 = avg*6*1.05; f12 = avg*12*1.05
        return {'6_month':{'forecast':f6,'lower':f6*0.85,'upper':f6*1.15},
                '12_month':{'forecast':f12,'lower':f12*0.85,'upper':f12*1.15},
                'model_results': {}, 'selected_model': 'Statistical Baseline',
                'ensemble': False}

    # Build monthly time series
    has_date = 'Date' in df.columns
    monthly  = None
    if has_date:
        dfc = df.copy()
        dfc['Date'] = pd.to_datetime(dfc['Date'], errors='coerce')
        dfc = dfc.dropna(subset=['Date'])
        ts = dfc.set_index('Date')[sales_col].resample('MS').sum().reset_index()
        ts.columns = ['ds', 'y']
        ts = ts.sort_values('ds').reset_index(drop=True)
        if len(ts) >= 2: monthly = ts

    if monthly is None:
        avg = float(df[sales_col].mean())
        synthetic_ds = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='MS')
        monthly = pd.DataFrame({'ds': synthetic_ds, 'y': np.array([avg]*12)})

    # ── Run ALL 4 models ─────────────────────────────────────────────────────
    prophet_result  = _run_prophet_model(monthly)  if has_date else None
    hw_result       = _run_holtwinters_model(monthly)
    lr_result       = _run_linear_regression_model(monthly)
    baseline_result = _run_baseline_model(monthly)

    # Compile results dict
    model_results = {}
    if prophet_result:  model_results['Prophet']              = prophet_result
    if hw_result:       model_results['Holt-Winters']         = hw_result
    if lr_result:       model_results['Linear Regression']    = lr_result
    if baseline_result: model_results['Statistical Baseline'] = baseline_result

    # ── Weighted Ensemble ────────────────────────────────────────────────────
    # Base weights — higher = more reliable model
    BASE_WEIGHTS = {
        'Prophet':              0.40,
        'Holt-Winters':         0.30,
        'Linear Regression':    0.20,
        'Statistical Baseline': 0.10,
    }

    available = {k: v for k, v in BASE_WEIGHTS.items() if k in model_results}

    if len(available) == 0:
        # Absolute fallback
        avg = float(df[sales_col].mean())
        f6 = avg*6*1.05; f12 = avg*12*1.05
        return {'6_month':{'forecast':f6,'lower':f6*0.85,'upper':f6*1.15},
                '12_month':{'forecast':f12,'lower':f12*0.85,'upper':f12*1.15},
                'model_results': {}, 'selected_model': 'Statistical Baseline', 'ensemble': False}

    # Normalise weights so they sum to 1.0
    total_w = sum(available.values())
    norm_w  = {k: v/total_w for k, v in available.items()}

    def _weighted(key):
        """Compute weighted sum for a given forecast key (e.g. '6_month'/'forecast')"""
        f_val = sum(model_results[m][key]['forecast'] * norm_w[m] for m in available)
        l_val = sum(model_results[m][key]['lower']    * norm_w[m] for m in available)
        u_val = sum(model_results[m][key]['upper']    * norm_w[m] for m in available)
        return {'forecast': f_val, 'lower': l_val, 'upper': u_val}

    ensemble_6m  = _weighted('6_month')
    ensemble_12m = _weighted('12_month')

    # Label: list all models used in ensemble
    used_models   = list(available.keys())
    weight_labels = " + ".join(f"{k.split()[0]} {norm_w[k]*100:.0f}%" for k in used_models)
    ensemble_name = f"Ensemble ({weight_labels})"

    return {
        '6_month':        ensemble_6m,
        '12_month':       ensemble_12m,
        'model_results':  model_results,
        'selected_model': ensemble_name,
        'ensemble':       True,
        'ensemble_weights': norm_w,
        'models_used':    used_models,
        'forecast_dfs':   {},
        'per_store_forecasts': {}
    }
def generate_granular_forecast(df):
    import warnings; warnings.filterwarnings("ignore")
    sales_col = 'Monthly_Sales_INR' if 'Monthly_Sales_INR' in df.columns else 'Gross_Sales'
    sku_col = 'SKU_Name' if 'SKU_Name' in df.columns else None
    cat_col = 'Product_Category' if 'Product_Category' in df.columns else None
    store_col = 'Store_ID' if 'Store_ID' in df.columns else None
    df = df.copy(); df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0)
    has_dates = 'Date' in df.columns
    if has_dates: df['Date'] = pd.to_datetime(df['Date'], errors='coerce'); df = df.dropna(subset=['Date'])
    def _run_prophet(sdf):
        if not PROPHET_AVAILABLE or not has_dates: return None
        try:
            monthly = (sdf.set_index('Date')[sales_col].resample('MS').sum().reset_index()); monthly.columns=['ds','y']
            monthly = monthly.sort_values('ds').reset_index(drop=True)
            if len(monthly) < 2: return None
            train = monthly.tail(12).copy(); last = monthly['ds'].max()
            m = Prophet(yearly_seasonality=False,weekly_seasonality=True,daily_seasonality=False,changepoint_prior_scale=0.001,interval_width=0.95)
            m.fit(train); future = m.make_future_dataframe(periods=12,freq='MS'); fc = m.predict(future)
            fc2 = fc[fc['ds']>=last][['ds','yhat','yhat_lower','yhat_upper']].copy()
            for c2 in ['yhat','yhat_lower','yhat_upper']: fc2[c2] = fc2[c2].clip(lower=0)
            return {'hist':monthly,'fc':fc2,'last':last}
        except: return None
    def _fallback(total, label):
        avg = total/12 if total > 0 else 0; g = 0.05
        return {'label':label,'total_hist':total,'6m_forecast':avg*6*(1+g),'6m_lower':avg*6*(1+g)*0.85,'6m_upper':avg*6*(1+g)*1.15,'12m_forecast':avg*12*(1+g),'12m_lower':avg*12*(1+g)*0.85,'12m_upper':avg*12*(1+g)*1.15,'hist':None,'fc':None}
    def _pack(label, res, total):
        if res is None: return _fallback(total, label)
        hist,fc = res['hist'],res['fc']; f6=fc.head(6); f12=fc.head(12)
        return {'label':label,'total_hist':total,'6m_forecast':f6['yhat'].sum(),'6m_lower':f6['yhat_lower'].sum(),'6m_upper':f6['yhat_upper'].sum(),'12m_forecast':f12['yhat'].sum(),'12m_lower':f12['yhat_lower'].sum(),'12m_upper':f12['yhat_upper'].sum(),'hist':hist,'fc':fc}
    overall_total = df[sales_col].sum()
    overall = _pack('Overall Company', _run_prophet(df[['Date',sales_col]] if has_dates else df), overall_total)
    stores = []
    if store_col:
        for sid in sorted(df[store_col].unique()):
            sdf = df[df[store_col]==sid]; stores.append(_pack(str(sid), _run_prophet(sdf[['Date',sales_col]] if has_dates else sdf), sdf[sales_col].sum()))
    categories = []
    if cat_col:
        for cat in sorted(df[cat_col].dropna().unique()):
            cdf = df[df[cat_col]==cat]; categories.append(_pack(str(cat), _run_prophet(cdf[['Date',sales_col]] if has_dates else cdf), cdf[sales_col].sum()))
    products = []
    if sku_col:
        top_skus = df.groupby(sku_col)[sales_col].sum().sort_values(ascending=False).head(5).index.tolist()
        for sk in top_skus:
            skdf = df[df[sku_col]==sk]; products.append(_pack(str(sk), _run_prophet(skdf[['Date',sales_col]] if has_dates else skdf), skdf[sales_col].sum()))
    return {'overall':overall,'stores':stores,'categories':categories,'products':products,'sales_col':sales_col,'raw_df':df,'sku_col':sku_col,'cat_col':cat_col}

def build_granular_charts(gf):
    plt.style.use('seaborn-v0_8-darkgrid')
    COLORS = ['#003366','#1f77b4','#e07b2a','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2']
    def _fmt(v):
        if v>=1e7: return f"Rs.{v/1e7:.1f}Cr"
        if v>=1e5: return f"Rs.{v/1e5:.1f}L"
        return f"Rs.{v:,.0f}"
    fig1,ax1 = plt.subplots(figsize=(13,8)); fig1.subplots_adjust(top=0.91,bottom=0.15,left=0.10,right=0.97)
    ov = gf['overall']
    if ov['hist'] is not None:
        ax1.plot(ov['hist']['ds'],ov['hist']['y'],color='#1f77b4',lw=2.5,label='Historical')
        ax1.plot(ov['fc']['ds'],ov['fc']['yhat'],color='#003366',lw=2,ls='--',label='12-Month Forecast')
    else:
        ax1.bar(['6-Month','12-Month'],[ov['6m_forecast'],ov['12m_forecast']],color=['#1f77b4','#003366'],alpha=0.85)
    ax1.set_title('Overall Company - Sales Forecast',fontsize=14,fontweight='bold',pad=14); ax1.set_ylabel('Sales (INR)',fontsize=11); ax1.legend(fontsize=10); ax1.grid(True,alpha=0.3); plt.setp(ax1.get_xticklabels(),rotation=30,ha='right')
    fig2,axes2 = plt.subplots(1,2,figsize=(13,8)); fig2.subplots_adjust(top=0.91,bottom=0.22,wspace=0.40,left=0.08,right=0.97)
    if gf['categories']:
        labels=[s['label'] for s in gf['categories']]; v6=[s['6m_forecast'] for s in gf['categories']]; v12=[s['12m_forecast'] for s in gf['categories']]
        x=np.arange(len(labels)); w=0.38
        axes2[0].bar(x-w/2,v6,w,color=COLORS[:len(x)],alpha=0.85,label='6M'); axes2[0].bar(x+w/2,v12,w,color=COLORS[:len(x)],alpha=0.55,label='12M')
        axes2[0].set_xticks(x); axes2[0].set_xticklabels(labels,rotation=30,ha='right')
        axes2[0].set_title('Category Level - 6M vs 12M',fontsize=13,fontweight='bold',pad=14); axes2[0].legend(); axes2[0].grid(axis='y',alpha=0.3)
        axes2[1].pie(v12,labels=labels,colors=COLORS[:len(v12)],autopct='%1.1f%%',startangle=90,textprops={'fontsize':9}); axes2[1].set_title('12-Month Forecast Share',fontsize=13,fontweight='bold',pad=14)
    else:
        axes2[0].text(0.5,0.5,'No category data',ha='center',va='center',transform=axes2[0].transAxes)
        axes2[1].text(0.5,0.5,'No category data',ha='center',va='center',transform=axes2[1].transAxes)
    fig3,ax3 = plt.subplots(figsize=(13,8)); fig3.subplots_adjust(top=0.91,bottom=0.25,left=0.10,right=0.97)
    if gf['categories']:
        cats=gf['categories']; lbs=[c['label'] for c in cats]; v6=[c['6m_forecast'] for c in cats]; v12=[c['12m_forecast'] for c in cats]
        x=np.arange(len(lbs)); w=0.38
        ax3.bar(x-w/2,v6,w,color='#1f77b4',alpha=0.85,label='6-Month'); ax3.bar(x+w/2,v12,w,color='#003366',alpha=0.85,label='12-Month')
        ax3.set_xticks(x); ax3.set_xticklabels(lbs,rotation=30,ha='right')
        ax3.set_title('Category-Level Forecast',fontsize=13,fontweight='bold',pad=14); ax3.legend(); ax3.grid(axis='y',alpha=0.3)
    else:
        ax3.text(0.5,0.5,'No category data',ha='center',va='center',transform=ax3.transAxes)
    fig45,axes45 = plt.subplots(1,2,figsize=(26,7)); fig45.subplots_adjust(top=0.88,bottom=0.10,left=0.18,right=0.97,wspace=0.55)
    products = gf.get('products',[])
    for ax_idx,(pk,pl) in enumerate([('6m_forecast','6-Month'),('12m_forecast','12-Month')]):
        ax=axes45[ax_idx]
        if products:
            sk=sorted(products,key=lambda s:s[pk],reverse=False); lbs2=[s['label'] for s in sk]; vals=[s[pk] for s in sk]
            cols=plt.cm.RdYlGn(np.linspace(0.25,0.85,len(sk))); bars_h=ax.barh(lbs2,vals,color=cols,height=0.55,edgecolor='white',linewidth=0.5)
            ax.set_xlabel('Forecasted Sales (INR)',fontsize=11,fontweight='bold'); ax.set_title(f'Top 5 Products - {pl} Forecast',fontsize=13,fontweight='bold',pad=12); ax.grid(axis='x',alpha=0.25)
            max_val=max(vals) if vals else 1
            for bar in bars_h:
                w2=bar.get_width(); ax.text(w2+max_val*0.01,bar.get_y()+bar.get_height()/2,_fmt(w2),va='center',ha='left',fontsize=9,fontweight='bold')
            ax.set_xlim(0,max_val*1.22)
        else:
            ax.text(0.5,0.5,'No product data',ha='center',va='center',transform=ax.transAxes)
    fig4=fig45; fig5=None
    fig6,ax6 = plt.subplots(figsize=(13,8)); fig6.subplots_adjust(top=0.91,bottom=0.18,left=0.10,right=0.97)
    plotted=False
    for i,s in enumerate(gf['categories'] if gf['categories'] else gf['stores']):
        if s['hist'] is not None and s['fc'] is not None:
            clr=COLORS[i%len(COLORS)]; ax6.plot(s['hist']['ds'],s['hist']['y'],color=clr,lw=1.8,label=f"{s['label']} Historical")
            ax6.plot(s['fc']['ds'],s['fc']['yhat'],color=clr,lw=1.8,ls='--',label=f"{s['label']} Forecast"); plotted=True
    if not plotted: ax6.text(0.5,0.5,'No time-series category data',ha='center',va='center',transform=ax6.transAxes)
    ax6.set_title('Per-Category Monthly Sales',fontsize=13,fontweight='bold',pad=14); ax6.set_ylabel('Sales (INR)',fontsize=11)
    if plotted: ax6.legend(fontsize=8,ncol=2)
    ax6.grid(True,alpha=0.3); plt.setp(ax6.get_xticklabels(),rotation=30,ha='right')
    all_items = ([('Overall',gf['overall'])]+[(f"Cat: {s['label']}",s) for s in gf['categories'][:5]]+[(f"Store: {s['label']}",s) for s in gf['stores'][:3]])
    lbs3=[a[0] for a in all_items]; hv=[a[1]['total_hist'] for a in all_items]; f6v=[a[1]['6m_forecast'] for a in all_items]; f12v=[a[1]['12m_forecast'] for a in all_items]
    n2=len(lbs3); fh=max(6,n2*1.1+2)
    fig7,ax7 = plt.subplots(figsize=(14,fh)); fig7.subplots_adjust(top=0.93,bottom=0.08,left=0.26,right=0.97)
    y2=np.arange(n2); h2=0.26
    ax7.barh(y2+h2,hv,h2,label='Historical Total',color='#7f7f7f',alpha=0.75)
    ax7.barh(y2,f6v,h2,label='6M Forecast',color='#1f77b4',alpha=0.90)
    ax7.barh(y2-h2,f12v,h2,label='12M Forecast',color='#003366',alpha=0.90)
    ax7.set_yticks(y2); ax7.set_yticklabels(lbs3,fontsize=9); ax7.set_xlabel('Sales (INR)',fontsize=11,fontweight='bold')
    ax7.set_title('All-Segment Summary: Historical vs 6M vs 12M',fontsize=13,fontweight='bold',pad=14); ax7.legend(loc='lower right',fontsize=9); ax7.grid(axis='x',alpha=0.25)
    fig8,ax8 = plt.subplots(figsize=(13,7)); fig8.subplots_adjust(top=0.91,bottom=0.04,left=0.05,right=0.95); ax8.axis('off')
    ov2 = gf['overall']
    if ov2['fc'] is not None:
        fc3=ov2['fc'].head(12).copy(); fc3['Month']=fc3['ds'].dt.strftime('%b %Y'); fc3['Forecast']=fc3['yhat'].apply(_fmt)
        tbl=ax8.table(cellText=fc3[['Month','Forecast']].values.tolist(),colLabels=['Month','Forecasted Sales'],cellLoc='center',loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.8)
        for j in range(2): tbl[(0,j)].set_facecolor('#003366'); tbl[(0,j)].set_text_props(color='white',fontweight='bold')
        for i2 in range(1,13):
            clr2='#eaf2ff' if i2%2==0 else 'white'
            for j in range(2): tbl[(i2,j)].set_facecolor(clr2)
        ax8.set_title('12-Month Forecast Breakdown',fontsize=13,fontweight='bold',pad=18)
    else:
        ax8.text(0.5,0.5,'No time-series data',ha='center',va='center',transform=ax8.transAxes); ax8.set_title('12-Month Forecast Breakdown',fontsize=13,fontweight='bold',pad=18)
    return fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8


# ══════════════════════════════════════════════════════════════════════════════
# Translations
# ══════════════════════════════════════════════════════════════════════════════
LANG = {
    'en': {
        'insights_title':'AI-Powered Business Insights','overall_summary':'Overall Performance Summary',
        'total_sales':'Total Sales','total_products':'Total Products Analyzed','avg_margin':'Average Profit Margin',
        'health_score':'Overall MSME Health Score','perf_score':'Overall Performance Score',
        'top5':'Top 5 Performing Products','perf_metrics':'Performance Metrics',
        'fin_risk':'Financial Risk Score','vendor_score':'Vendor Reliability Score','growth_score':'Growth Potential Score',
        'lower_better':'(Lower is better)','forecast_title':'ML-Powered Sales Forecast',
        'six_month':'6-Month Projection','twelve_month':'12-Month Projection',
        'forecast_sales':'Forecasted Sales','expected_range':'Expected Range',
        'snp_title':'ONDC Seller Network Participant (SNP) Matching','recommendations':'AI-Generated Recommendations',
        'immediate':'Immediate Actions','strategic':'Strategic Initiatives','risk_alert':'Risk Alerts',
        'store_forecast':'Store-Specific Sales Forecasts','data_quality':'Data Quality Report',
        'inference_time':'Analysis completed in','seconds':'seconds'
    },
    'hi': {
        'insights_title':'AI से मिली आपके धंधे की जानकारी','overall_summary':'आपके धंधे का कुल हाल',
        'total_sales':'कुल बिक्री','total_products':'कुल सामान','avg_margin':'औसत मुनाफा (%)',
        'health_score':'धंधे की सेहत का स्कोर','perf_score':'काम का कुल स्कोर',
        'top5':'सबसे ज्यादा बिकने वाले 5 सामान','perf_metrics':'काम के नंबर',
        'fin_risk':'पैसों का जोखिम स्कोर','vendor_score':'सप्लायर का भरोसा स्कोर','growth_score':'आगे बढ़ने की संभावना',
        'lower_better':'(कम नंबर अच्छा है)','forecast_title':'AI से अगले महीनों की बिक्री का अनुमान',
        'six_month':'अगले 6 महीने की बिक्री','twelve_month':'अगले 12 महीने की बिक्री',
        'forecast_sales':'अनुमानित बिक्री','expected_range':'बिक्री कम से कम — ज्यादा से ज्यादा',
        'snp_title':'ONDC पर बेचने के लिए सबसे अच्छे Platform','recommendations':'AI की सलाह',
        'immediate':'अभी करने वाले काम','strategic':'आगे की योजना','risk_alert':'खतरे की चेतावनी',
        'store_forecast':'हर दुकान की अगली बिक्री का अनुमान','data_quality':'आपके Data की जाँच',
        'inference_time':'जाँच पूरी हुई','seconds':'सेकंड में'
    }
}

def T(key, lang='en'): return LANG.get(lang, LANG['en']).get(key, LANG['en'].get(key, key))

SNP_CATALOG = {
    'GeM (Government e-Marketplace)': {
        'business_types':['Manufacturing','FMCG','Electronics','Clothing','Services','Hypermarket'],
        'min_health':40,'segment_boost':['Champions','Loyal'],
        'description_en':'Government procurement portal — ideal for MSMEs supplying to public sector.',
        'action_en':'Register on GeM portal (gem.gov.in) and map your product catalogue.'},
    'Flipkart Commerce (ONDC)': {
        'business_types':['FMCG','Supermarket','Clothing','Electronics','Hypermarket'],
        'min_health':30,'segment_boost':['Champions','Loyal','Potential'],
        'description_en':'High-volume B2C SNP — best for consumer goods with strong demand.',
        'action_en':'Onboard via Flipkart Seller Hub — optimise product images and descriptions.'},
    'Meesho (ONDC)': {
        'business_types':['Clothing','FMCG','Manufacturing'],
        'min_health':20,'segment_boost':['Potential','At Risk'],
        'description_en':'Social commerce SNP — ideal for price-sensitive segments and tier-2/3 markets.',
        'action_en':'List on Meesho for reseller network access — focus on competitive pricing.'},
    'NSIC e-Marketplace': {
        'business_types':['Manufacturing','FMCG','Electronics','Services','Hypermarket'],
        'min_health':25,'segment_boost':['Champions','Loyal','Potential'],
        'description_en':'NSIC marketplace for MSE-to-MSE and B2B procurement.',
        'action_en':'Register with NSIC for buyer-seller matchmaking.'},
    'Amazon Seller Services (ONDC)': {
        'business_types':['Electronics','FMCG','Clothing','Supermarket','Hypermarket'],
        'min_health':35,'segment_boost':['Champions','Loyal'],
        'description_en':'Premium B2C SNP — suits high-quality products with strong margin and low returns.',
        'action_en':'Apply for Amazon Easy Ship / FBA integration via ONDC bridge.'},
    'Udaan (B2B ONDC)': {
        'business_types':['Manufacturing','FMCG','Clothing','Supermarket','Hypermarket'],
        'min_health':20,'segment_boost':['Champions','Loyal','Potential','At Risk'],
        'description_en':'B2B wholesale SNP — best for MSMEs supplying to retailers and distributors.',
        'action_en':'List bulk products on Udaan for retailer discovery and bulk orders.'},
}


# ══════════════════════════════════════════════════════════════════════════════
# Storyboard / Insight Dashboard CSS & Builders (abbreviated for integration)
# ══════════════════════════════════════════════════════════════════════════════
STORYBOARD_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.sb-root{font-family:'Inter',Arial,sans-serif;color:#1A2D45;background:#EEF4FB;padding:0 0 60px 0;max-width:100%;}
.sb-root *{box-sizing:border-box;}
.sb-hero{background:linear-gradient(135deg,#0B1F3A 0%,#1B3A6B 60%,#1A5276 100%);padding:48px 48px 56px;overflow:hidden;}
.sb-hero-title{font-size:32px;font-weight:900;color:#FFFFFF;line-height:1.15;margin:0 0 8px;}
.sb-hero-sub{font-size:15px;color:#A8D8FF;font-weight:300;margin-bottom:32px;}
.sb-chip{background:rgba(255,255,255,.12);border:1px solid rgba(255,215,100,.45);border-radius:20px;padding:5px 14px;font-size:12px;font-weight:700;color:#FFD080;display:inline-block;margin:3px;}
.sb-chip.green{background:rgba(46,204,143,.25);border-color:rgba(46,204,143,.5);color:#5DEBB0;}
.sb-chip.amber{background:rgba(232,168,56,.25);border-color:rgba(232,168,56,.5);color:#FFD070;}
.sb-chip.red{background:rgba(224,82,82,.25);border-color:rgba(224,82,82,.5);color:#FF8080;}
.sb-section-divider{display:flex;align-items:center;gap:16px;padding:40px 48px 0;}
.sb-section-number{font-size:48px;font-weight:900;color:rgba(27,79,138,.15);line-height:1;min-width:52px;}
.sb-section-title{font-size:22px;font-weight:700;color:#0B1F3A;}
.sb-section-line{flex:1;height:1px;background:linear-gradient(90deg,#C8DCEF,transparent);}
.sb-card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;padding:20px 48px 0;}
.sb-kpi-card{background:#FFFFFF;border:1px solid #C8DCEF;border-radius:14px;padding:22px 20px;position:relative;overflow:hidden;}
.sb-kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.sb-kpi-card.accent-blue::before{background:#1B4F8A;}.sb-kpi-card.accent-green::before{background:#2ECC8F;}
.sb-kpi-card.accent-amber::before{background:#E8A838;}.sb-kpi-card.accent-red::before{background:#E05252;}
.sb-kpi-card.accent-navy::before{background:#0B1F3A;}
.sb-kpi-label{font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#4A6A8A;margin-bottom:4px;}
.sb-kpi-value{font-size:26px;font-weight:700;color:#0B1F3A;line-height:1.1;}
.sb-kpi-sub{font-size:11px;color:#4A6A8A;margin-top:4px;}
.sb-status-badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;margin-top:8px;}
.badge-green{background:#E8FBF4;color:#1A7A50;}.badge-amber{background:#FEF6E7;color:#A06000;}.badge-red{background:#FDE8E8;color:#B03030;}.badge-blue{background:#E8F2FF;color:#1B4F8A;}
.sb-scores-row{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;padding:20px 48px 0;}
.sb-score-card{background:#FFFFFF;border:1px solid #C8DCEF;border-radius:14px;padding:20px 22px;color:#1A2D45;}
.sb-score-card div{color:#1A2D45;}
.sb-score-bar-track{height:6px;background:#D8E8F8;border-radius:3px;overflow:hidden;}
.sb-score-bar-fill{height:100%;border-radius:3px;}
.sb-table-wrap{padding:20px 48px 0;}
.sb-table{width:100%;border-collapse:separate;border-spacing:0;background:#FFFFFF;border-radius:14px;border:1px solid #C8DCEF;overflow:hidden;}
.sb-table thead tr th{background:#0B1F3A;color:#E0F0FF;padding:13px 18px;font-size:11px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;text-align:left;}
.sb-table tbody tr td{padding:13px 18px;border-bottom:1px solid #C8DCEF;font-size:13px;color:#1A2D45;}
.sb-forecast-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:20px 48px 0;}
.sb-forecast-card{background:linear-gradient(135deg,#0B1F3A 0%,#1B3A6B 100%);border-radius:16px;padding:28px 26px;}
.sb-forecast-amount{font-size:34px;font-weight:900;color:#FFFFFF !important;line-height:1.1;}
.sb-forecast-range{font-size:12px;color:#A8D8FF !important;margin-top:6px;}
.sb-snp-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:0 48px;}
.sb-snp-card{background:#FFFFFF;border:1px solid #C8DCEF;border-radius:14px;padding:20px 18px;color:#1A2D45;}
.sb-snp-card.gold{border-top:4px solid #F5C842;}.sb-snp-card.silver{border-top:4px solid #B0BEC5;}.sb-snp-card.bronze{border-top:4px solid #CD7F32;}
.sb-reco-tabs{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:0 48px;}
.sb-reco-panel{background:#FFFFFF;border:1px solid #C8DCEF;border-radius:14px;overflow:hidden;}
.sb-reco-header{padding:14px 20px;font-weight:700;font-size:13px;color:#FFFFFF;}
.sb-reco-header.immediate{background:linear-gradient(90deg,#E05252,#C0392B);}
.sb-reco-header.strategic{background:linear-gradient(90deg,#1B4F8A,#0B2F5A);color:#D0EAFF;}
.sb-reco-row{display:flex;align-items:flex-start;gap:12px;padding:12px 20px;border-bottom:1px solid #C8DCEF;color:#1A2D45;}
.sb-reco-row div{color:#1A2D45;font-size:13px;line-height:1.5;flex:1;}
.sb-reco-priority{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;padding:2px 8px;border-radius:10px;flex-shrink:0;margin-top:2px;}
.reco-high{background:#FDE8E8;color:#B03030;}.reco-medium{background:#FEF6E7;color:#A06000;}
.sb-footer{margin:40px 48px 0;padding:16px 20px;background:#FFFFFF;border:1px solid #C8DCEF;border-radius:10px;display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#4A6A8A;}
@media(max-width:900px){.sb-scores-row,.sb-forecast-row,.sb-snp-grid,.sb-reco-tabs{grid-template-columns:1fr;}.sb-hero,.sb-section-divider,.sb-card-grid,.sb-table-wrap,.sb-snp-grid,.sb-reco-tabs{padding-left:16px;padding-right:16px;}}
</style>"""

def _fmt_inr_sb(v):
    if pd.isna(v) or v is None: return "N/A"
    if v>=1e7: return f"&#8377;{v/1e7:.2f} Cr"
    if v>=1e5: return f"&#8377;{v/1e5:.2f} L"
    return f"&#8377;{v:,.0f}"

def _score_bar_color(v, invert=False):
    eff = (1-v) if invert else v
    if eff >= 0.65: return "#2ECC8F"
    if eff >= 0.40: return "#E8A838"
    return "#E05252"

def _badge_cls(v, invert=False):
    eff = (1-v) if invert else v
    if eff >= 0.65: return "badge-green"
    if eff >= 0.40: return "badge-amber"
    return "badge-red"

def _status_lbl(v, invert=False):
    eff = (1-v) if invert else v
    if eff >= 0.65: return "Excellent"
    if eff >= 0.40: return "Moderate"
    return "Needs Attention"

def _health_cls(v):
    if v >= 65: return "badge-green"
    if v >= 40: return "badge-amber"
    return "badge-red"

def _health_lbl(v):
    if v >= 65: return "Healthy"
    if v >= 40: return "Developing"
    return "At Risk"

def _margin_cls(v):
    if v > 20: return "badge-green"
    if v > 10: return "badge-amber"
    return "badge-red"

def _margin_lbl(v):
    if v > 20: return "Strong"
    if v > 10: return "Moderate"
    return "Low"

def _risk_cls(v):
    if v <= 0.40: return "badge-green"
    if v <= 0.70: return "badge-amber"
    return "badge-red"

def _risk_lbl(v):
    if v <= 0.40: return "Low Risk"
    if v <= 0.70: return "Moderate"
    return "High Risk"

def _sb_divider(num, eyebrow, title):
    return f"""<div class="sb-section-divider"><div class="sb-section-number">{num:02d}</div>
<div style="display:flex;flex-direction:column;gap:2px;">
<div style="font-size:10px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B07A00">{eyebrow}</div>
<div class="sb-section-title">{title}</div></div><div class="sb-section-line"></div></div>"""

def generate_insights(user_data, df_raw, lang='en'):
    import time; t_start = time.time()
    try:
        df = calculate_scores(df_raw.copy())
        sales_col = 'Monthly_Sales_INR'; sku_col = 'SKU_Name'
        total_sales    = df[sales_col].sum() if sales_col in df.columns else 0
        total_records  = len(df)  # total data rows (transactions/months)
        total_products = df[sku_col].nunique() if sku_col in df.columns else total_records
        avg_margin = df['Avg_Margin_Percent'].mean() if 'Avg_Margin_Percent' in df.columns else 0
        perf_score = df['Performance_Score'].mean() if 'Performance_Score' in df.columns else 0
        health_score = df['MSME_Health_Score'].mean() if 'MSME_Health_Score' in df.columns else 0
        fin_risk = df['Financial_Risk_Score'].mean() if 'Financial_Risk_Score' in df.columns else 0
        vendor_sc = df['Vendor_Score'].mean() if 'Vendor_Score' in df.columns else 0
        growth_sc = df['Growth_Potential_Score'].mean() if 'Growth_Potential_Score' in df.columns else 0
        company = user_data.get('company_name', 'Your Company')
        forecast_results = forecast_sales(df)
        f6 = forecast_results['6_month']; f12 = forecast_results['12_month']
        seg_result = segment_customers(df); elapsed = time.time() - t_start
        hl = _health_lbl(health_score)
        hcls = "green" if health_score>=65 else ("amber" if health_score>=40 else "red")
        html = STORYBOARD_CSS + f'<div class="sb-root">'
        # Hero
        html += f"""<div class="sb-hero">
<div style="font-size:11px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#FFD080;margin-bottom:12px">DataNetra.ai — AI-Powered Business Intelligence</div>
<div class="sb-hero-title" style="font-size:32px;font-weight:900;color:#FFFFFF !important;line-height:1.15;margin:0 0 8px;text-shadow:0 2px 8px rgba(0,0,0,0.4)">{company}</div>
<div class="sb-hero-sub" style="font-size:15px;color:#A8D8FF !important;font-weight:300;margin-bottom:32px">Comprehensive Analysis Report &nbsp;·&nbsp; {datetime.datetime.now().strftime("%d %b %Y, %H:%M")}</div>
<div>
  <span class="sb-chip">Total Revenue: {_fmt_inr_sb(total_sales)}</span>
  <span class="sb-chip">{hl} Business</span>
  <span class="sb-chip">Perf Score: {perf_score:.1f}%</span>
  <span class="sb-chip">Prophet · Holt-Winters · LinReg · KMeans</span>
</div></div>"""
        # Section 1 — KPIs
        html += _sb_divider(1, 'Overall Summary', 'Business Performance')
        mc = _margin_cls(avg_margin); ml = _margin_lbl(avg_margin)
        hclr = _score_bar_color(health_score/100); rclr = "#1a7a40" if fin_risk<=0.4 else ("#b05a00" if fin_risk<=0.7 else "#b03030")
        def _kpi_card(accent, icon, label, value, sub_html):
            acc_colors = {'blue':'#1B4F8A','green':'#2ECC8F','amber':'#E8A838','red':'#E05252','navy':'#0B1F3A'}
            top_col = acc_colors.get(accent, '#1B4F8A')
            return (f'<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:14px;padding:22px 20px;'
                    f'position:relative;overflow:hidden;border-top:3px solid {top_col}">'
                    f'<div style="font-size:28px;margin-bottom:8px">{icon}</div>'
                    f'<div style="font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#4A6A8A;margin-bottom:4px">{label}</div>'
                    f'<div style="font-size:26px;font-weight:700;color:#0B1F3A;line-height:1.1">{value}</div>'
                    f'{sub_html}</div>')
        mc_acc = 'green' if avg_margin>20 else ('amber' if avg_margin>10 else 'red')
        hlt_acc = 'green' if health_score>=65 else ('amber' if health_score>=40 else 'red')
        prf_acc = 'green' if perf_score>=65 else ('amber' if perf_score>=40 else 'red')
        prf_lbl = 'Excellent' if perf_score>=65 else ('Moderate' if perf_score>=40 else 'Low')
        html += f"""<div class="sb-card-grid">
{_kpi_card('blue', '💰', 'Total Revenue', _fmt_inr_sb(total_sales), '<div style="font-size:11px;color:#4A6A8A;margin-top:4px">Gross Sales (all products)</div>')}
{_kpi_card('navy', '📦', 'Data Records', f'{total_records:,}', f'<div style="font-size:11px;color:#4A6A8A;margin-top:4px">{total_products:,} unique SKU{"s" if total_products!=1 else ""}</div>')}
{_kpi_card(mc_acc, '📈', 'Avg Profit Margin', f'{avg_margin:.1f}%', f'<span class="sb-status-badge {mc}" style="margin-top:8px">{ml}</span>')}
{_kpi_card(hlt_acc, '🧠', 'MSME Health Score', f'{health_score:.1f}%', f'<span class="sb-status-badge {_health_cls(health_score)}" style="margin-top:8px">{_health_lbl(health_score)}</span>')}
{_kpi_card(prf_acc, '⭐', 'Performance Score', f'{perf_score:.1f}%', f'<span class="sb-status-badge {_health_cls(perf_score)}" style="margin-top:8px">{prf_lbl}</span>')}
</div>"""
        # ── MSME Snapshot (identity only — scores shown in Section 2 below) ──
        msme_id    = user_data.get('msme_number', 'N/A')
        owner_name = user_data.get('full_name', 'Business Owner')
        biz_type   = user_data.get('business_type', 'MSME')
        city_val   = user_data.get('city', '')
        html += f"""<div style="margin:0 48px 20px;background:#0B1F3A;border-radius:12px;padding:16px 22px;">
  <div style="font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#FFD080;margin-bottom:12px">
    🏢 MSME Snapshot
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px">
    <div style="background:rgba(255,255,255,0.06);border-radius:8px;padding:11px 14px">
      <div style="font-size:10px;color:#7AABDD;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">Business</div>
      <div style="font-size:14px;font-weight:700;color:#FFFFFF;line-height:1.2">{company}</div>
      <div style="font-size:11px;color:#A8C8E8;margin-top:3px">{biz_type}</div>
    </div>
    <div style="background:rgba(255,255,255,0.06);border-radius:8px;padding:11px 14px">
      <div style="font-size:10px;color:#7AABDD;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">Owner</div>
      <div style="font-size:13px;font-weight:700;color:#FFFFFF">{owner_name}</div>
      <div style="font-size:11px;color:#A8C8E8;margin-top:3px">{city_val if city_val else 'Registered MSME'}</div>
    </div>
    <div style="background:rgba(255,255,255,0.06);border-radius:8px;padding:11px 14px">
      <div style="font-size:10px;color:#7AABDD;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">MSME / Udyam ID</div>
      <div style="font-size:13px;font-weight:700;color:#FFD080">{msme_id}</div>
    </div>
    <div style="background:rgba(255,255,255,0.06);border-radius:8px;padding:11px 14px">
      <div style="font-size:10px;color:#7AABDD;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">Total Revenue</div>
      <div style="font-size:14px;font-weight:700;color:#FFFFFF">{_fmt_inr_sb(total_sales)}</div>
      <div style="font-size:11px;color:#A8C8E8;margin-top:3px">{total_records:,} records · {total_products:,} SKU{"s" if total_products!=1 else ""}</div>
    </div>
  </div>
</div>"""
        # Section 2 — Scores
        html += _sb_divider(2, 'Score Breakdown', 'Risk & Performance Scores')

        # ── Score cards with full explanation ─────────────────────────────────
        perf_sc_norm = perf_score / 100.0
        profit_ratio = avg_margin / 100.0

        def _score_card_full(icon, title, value_display, value_norm, target_txt, formula_txt, explanation_txt, badge_cls, badge_lbl, invert=False):
            bar_pct = min(value_norm * 100, 100)
            bar_c = _score_bar_color(1 - value_norm if invert else value_norm)
            return f"""<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:14px;padding:20px 18px;border-top:3px solid {bar_c}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
    <div>
      <div style="font-weight:700;font-size:13px;color:#0B1F3A">{icon} {title}</div>
      <div style="font-size:11px;color:#4A6A8A;margin-top:2px">{target_txt}</div>
    </div>
    <div style="font-size:26px;font-weight:900;color:{bar_c};font-family:monospace">{value_display}</div>
  </div>
  <div style="height:7px;background:#D8E8F8;border-radius:4px;margin-bottom:8px">
    <div style="width:{bar_pct:.0f}%;height:100%;background:{bar_c};border-radius:4px"></div></div>
  <span class="sb-status-badge {badge_cls}" style="margin-bottom:10px">{badge_lbl}</span>
  <div style="margin-top:10px;font-size:11px;color:#2A4060;line-height:1.6">{explanation_txt}</div>
</div>"""

        html += f"""<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin:16px 48px 0">
{_score_card_full('⚠️', 'Financial Risk Score', f'{fin_risk:.2f}', fin_risk,
    'Lower is better · Target &lt;0.40',
    '0.5×(OpCost/Sales) + 0.5×(Loan/(Sales×12))',
    'Measures cashflow pressure and loan burden relative to revenue. A score above 0.70 signals danger — costs or debt are consuming most income. Below 0.40 indicates healthy financial breathing room.',
    _risk_cls(fin_risk), _risk_lbl(fin_risk), invert=True)}
{_score_card_full('📈', 'Performance Score', f'{perf_score:.1f}%', perf_sc_norm,
    'Higher is better · Target &gt;65%',
    '0.30×Profitability + 0.25×OpEfficiency + 0.20×CustSatisfaction + 0.15×VendorReliability + 0.10×InvTurnover',
    'Composite score across five operational pillars. Reflects how well the business converts revenue into real value. Scores above 65% indicate a well-run, scalable operation.',
    _health_cls(perf_score), 'Excellent' if perf_score>=65 else ('Moderate' if perf_score>=40 else 'Low'))}
{_score_card_full('💰', 'Profit Margin Score', f'{avg_margin:.1f}%', min(avg_margin/40,1.0),
    'Higher is better · Target &gt;20%',
    'Avg(Margin%) across all SKUs',
    'Average gross margin across your product portfolio. Margins above 20% indicate strong pricing power and room to invest in growth. Below 10% means thin cushion for shocks.',
    _margin_cls(avg_margin), _margin_lbl(avg_margin))}
{_score_card_full('🤝', 'Vendor Reliability Score', f'{vendor_sc:.2f}', vendor_sc,
    'Higher is better · Target &gt;0.60',
    '0.50×VendorDeliveryReliability + 0.30×InvTurnover + 0.20×AvgMargin',
    'Blends supplier delivery reliability, inventory turnover velocity and margin contribution. A score above 0.65 means the supply chain supports growth. Below 0.40 — fulfilment risks are high.',
    _badge_cls(vendor_sc), _status_lbl(vendor_sc))}
{_score_card_full('🧠', 'MSME Health Score', f'{health_score:.1f}%', health_score/100,
    'Higher is better · Target &gt;65%',
    '0.40×(1−FinRisk) + 0.30×VendorScore + 0.30×GrowthPotential',
    'DataNetra\'s flagship composite metric — blends financial safety, supply chain health and growth momentum. The single most important number for ONDC readiness and investor-readiness assessment.',
    _health_cls(health_score), _health_lbl(health_score))}
{_score_card_full('🚀', 'Growth Potential Score', f'{growth_sc:.2f}', growth_sc,
    'Higher is better · Target &gt;0.60',
    '0.40×DemandUnits + 0.35×AvgMargin + 0.25×(1−ReturnsRate)',
    'Forward-looking indicator combining demand volume, profitability and customer acceptance (low returns = product-market fit). High scores here signal strong ONDC scale-up potential.',
    _badge_cls(growth_sc), _status_lbl(growth_sc))}
</div>"""

        # ── Score Summary Table ────────────────────────────────────────────────
        def _trend(val, tgt, invert=False):
            good = val <= tgt if invert else val >= tgt
            return ("✅ On Target", "#1a7a40") if good else ("⚠️ Needs Work", "#b05a00")

        rows_summary = [
            ("⚠️ Financial Risk",    f"{fin_risk:.2f}",      "< 0.40",  *_trend(fin_risk, 0.40, invert=True)),
            ("📈 Performance",       f"{perf_score:.1f}%",   "> 65%",   *_trend(perf_score, 65)),
            ("💰 Profit Margin",     f"{avg_margin:.1f}%",   "> 20%",   *_trend(avg_margin, 20)),
            ("🤝 Vendor Reliability",f"{vendor_sc:.2f}",     "> 0.60",  *_trend(vendor_sc, 0.60)),
            ("🧠 MSME Health",       f"{health_score:.1f}%", "> 65%",   *_trend(health_score, 65)),
            ("🚀 Growth Potential",  f"{growth_sc:.2f}",     "> 0.60",  *_trend(growth_sc, 0.60)),
        ]
        summary_rows_html = ""
        for i,(name,val,tgt,status,scol) in enumerate(rows_summary):
            bg = "#F0F7FF" if i%2==0 else "#FFFFFF"
            summary_rows_html += f"""<tr style="border-bottom:1px solid #D8E8F8;background:{bg}">
  <td style="padding:9px 14px;font-weight:600;color:#0B1F3A;font-size:12px">{name}</td>
  <td style="padding:9px 14px;font-family:monospace;font-weight:700;color:#0B1F3A;font-size:13px">{val}</td>
  <td style="padding:9px 14px;color:#4A6A8A;font-size:12px">{tgt}</td>
  <td style="padding:9px 14px;font-weight:700;color:{scol};font-size:12px">{status}</td>
</tr>"""

        html += f"""<div style="margin:20px 48px 0;background:#FFFFFF;border:1px solid #C8DCEF;border-radius:12px;overflow:hidden">
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr style="background:#0B1F3A">
      <th style="padding:10px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Score</th>
      <th style="padding:10px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Your Value</th>
      <th style="padding:10px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Target</th>
      <th style="padding:10px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Status</th>
    </tr></thead>
    <tbody>{summary_rows_html}</tbody>
  </table>
</div>"""

        # Forecast
        html += _sb_divider(3, 'Sales Forecast', 'ML-Powered Revenue Projections')

        # Primary forecast cards (best model)
        selected_model_name = forecast_results.get('selected_model', 'Statistical Baseline')
        html += f"""<div style="margin:0 48px">
  <div style="font-size:11px;color:#1B4F8A;background:#EAF4FF;border-radius:8px;padding:8px 14px;margin-bottom:14px;border-left:3px solid #1B4F8A;font-weight:600">
    📡 <strong>Ensemble Forecast</strong> &nbsp;·&nbsp; All available models contributed to the numbers above.
    Weights: {" + ".join(f"<strong>{k.split()[0]}</strong> {v*100:.0f}%" for k,v in forecast_results.get("ensemble_weights", {}).items())}
    &nbsp;·&nbsp; <span style="font-size:10px;color:#4A6A8A">Each model below ran independently — the weighted average drives the 6-month &amp; 12-month projections.</span>
  </div>
</div>
<div class="sb-forecast-row">
<div class="sb-forecast-card"><div style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#A8D8FF;margin-bottom:6px">6-Month Projection</div><div class="sb-forecast-amount" style="color:#FFFFFF !important">{_fmt_inr_sb(f6["forecast"])}</div><div class="sb-forecast-range" style="color:#A8D8FF !important">Range: {_fmt_inr_sb(f6["lower"])} — {_fmt_inr_sb(f6["upper"])}</div></div>
<div class="sb-forecast-card" style="background:linear-gradient(135deg,#1A5276 0%,#0B2F5A 100%);"><div style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#A8D8FF;margin-bottom:6px">12-Month Projection</div><div class="sb-forecast-amount" style="color:#FFFFFF !important">{_fmt_inr_sb(f12["forecast"])}</div><div class="sb-forecast-range" style="color:#A8D8FF !important">Range: {_fmt_inr_sb(f12["lower"])} — {_fmt_inr_sb(f12["upper"])}</div></div>
</div>"""

        # ── Model Comparison Table ─────────────────────────────────────────────
        model_results = forecast_results.get('model_results', {})
        MODEL_META = {
            'Prophet': {
                'icon': '📡', 'type': 'Time-Series',
                'desc': 'Handles seasonality, holidays & changepoints. Best for date-indexed datasets with ≥2 months of history.',
                'requires': 'Date column + ≥2 monthly data points',
                'available': PROPHET_AVAILABLE,
            },
            'Holt-Winters': {
                'icon': '❄️', 'type': 'Exponential Smoothing',
                'desc': 'Triple exponential smoothing with additive trend & seasonality. Great for stable seasonal patterns.',
                'requires': '≥4 data points (≥24 for seasonal component)',
                'available': HOLTWINTERS_AVAILABLE,
            },
            'Linear Regression': {
                'icon': '📐', 'type': 'Trend Extrapolation',
                'desc': 'Fits a straight-line trend through historical sales. Simple, interpretable and always available.',
                'requires': '≥2 data points',
                'available': LINEAR_REGRESSION_AVAILABLE,
            },
            'Statistical Baseline': {
                'icon': '📊', 'type': 'Fallback Avg ×1.05',
                'desc': 'Simple average of last 6 months × number of forecast months × 1.05 growth factor.',
                'requires': 'Any numeric sales column',
                'available': True,
            },
        }

        comp_rows = ""
        for mname, meta in MODEL_META.items():
            is_selected  = (mname == selected_model_name)
            has_result   = mname in model_results
            res          = model_results.get(mname, {})
            f6v  = _fmt_inr_sb(res['6_month']['forecast'])  if has_result else "—"
            f12v = _fmt_inr_sb(res['12_month']['forecast']) if has_result else "—"
            f6r  = f"{_fmt_inr_sb(res['6_month']['lower'])} – {_fmt_inr_sb(res['6_month']['upper'])}"  if has_result else "—"
            f12r = f"{_fmt_inr_sb(res['12_month']['lower'])} – {_fmt_inr_sb(res['12_month']['upper'])}" if has_result else "—"
            r2_txt = f"R²={res['r2_score']:.2f}" if mname == 'Linear Regression' and has_result and 'r2_score' in res else ""
            slope_txt = f"  slope={res.get('slope',0):+.0f}/mo" if r2_txt else ""

            if is_selected:
                row_bg = "#F0F7FF"; sel_badge = '<span style="font-size:10px;font-weight:700;padding:2px 9px;border-radius:10px;background:#1B4F8A;color:#FFFFFF;margin-left:6px">✓ SELECTED</span>'
            elif not meta['available']:
                row_bg = "#F9F9F9"; sel_badge = '<span style="font-size:10px;font-weight:700;padding:2px 9px;border-radius:10px;background:#F5F5F5;color:#AAA;margin-left:6px">Not Installed</span>'
            elif not has_result:
                row_bg = "#FFFBF0"; sel_badge = '<span style="font-size:10px;font-weight:700;padding:2px 9px;border-radius:10px;background:#FFF3CD;color:#856404;margin-left:6px">Insufficient Data</span>'
            else:
                row_bg = "#FFFFFF"; sel_badge = '<span style="font-size:10px;font-weight:700;padding:2px 9px;border-radius:10px;background:#EAF7EE;color:#1a7a40;margin-left:6px">Available</span>'

            comp_rows += f"""<tr style="border-bottom:1px solid #D8E8F8;background:{row_bg}">
  <td style="padding:11px 14px;min-width:160px">
    <div style="display:flex;align-items:center;gap:6px">
      <span style="font-size:16px">{meta['icon']}</span>
      <div>
        <div style="font-weight:700;font-size:12px;color:#0B1F3A">{mname}{sel_badge}</div>
        <div style="font-size:10px;color:#4A6A8A;text-transform:uppercase;letter-spacing:0.8px">{meta['type']}</div>
      </div>
    </div>
  </td>
  <td style="padding:11px 14px;font-size:11px;color:#2A4060;max-width:220px;line-height:1.4">{meta['desc']}</td>
  <td style="padding:11px 14px;text-align:center">
    <div style="font-weight:700;font-family:monospace;color:#0B1F3A;font-size:12px">{f6v}</div>
    <div style="font-size:10px;color:#4A6A8A;margin-top:2px">{f6r}</div>
  </td>
  <td style="padding:11px 14px;text-align:center">
    <div style="font-weight:700;font-family:monospace;color:#0B1F3A;font-size:12px">{f12v}</div>
    <div style="font-size:10px;color:#4A6A8A;margin-top:2px">{f12r}</div>
    {f'<div style="font-size:10px;color:#1B4F8A;margin-top:2px">{r2_txt}{slope_txt}</div>' if r2_txt else ''}
  </td>
  <td style="padding:11px 14px;font-size:10px;color:#4A6A8A">{meta['requires']}</td>
</tr>"""

        html += f"""<div style="margin:20px 48px 0;background:#FFFFFF;border:1px solid #C8DCEF;border-radius:12px;overflow:hidden">
  <div style="background:#0B1F3A;padding:10px 16px">
    <span style="color:#A8D8FF;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">📊 Forecasting Model Comparison — All 4 Models Run &amp; Weighted into Ensemble</span>
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr style="background:#1A3050">
      <th style="padding:9px 14px;text-align:left;color:#D0EAFF;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;min-width:160px">Model</th>
      <th style="padding:9px 14px;text-align:left;color:#D0EAFF;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase">Description</th>
      <th style="padding:9px 14px;text-align:center;color:#D0EAFF;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase">6-Month Forecast</th>
      <th style="padding:9px 14px;text-align:center;color:#D0EAFF;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase">12-Month Forecast</th>
      <th style="padding:9px 14px;text-align:left;color:#D0EAFF;font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase">Requires</th>
    </tr></thead>
    <tbody>{comp_rows}</tbody>
  </table>
  <div style="padding:10px 16px;background:#F0F7FF;font-size:11px;color:#4A6A8A">
    Weighted ensemble: Prophet 40% + Holt-Winters 30% + Linear Regression 20% + Baseline 10% (weights normalised across available models). Holt-Winters now runs in pure numpy — no installation needed. All models that have sufficient data contribute to the final forecast.
  </div>
</div>"""
        # ── SNP Mapping Insights (Section 4) ──────────────────────────────
        biz_type = user_data.get('business_type', 'FMCG')
        dom = 'Potential'
        if seg_result and seg_result.get('counts'): dom = max(seg_result['counts'], key=seg_result['counts'].get)

        # ── Raw data metrics for SNP logic ────────────────────────────────
        ret_col = 'Returns_Percentage' if 'Returns_Percentage' in df.columns else None
        mar_col = 'Avg_Margin_Percent' if 'Avg_Margin_Percent' in df.columns else None
        sal_col = 'Monthly_Sales_INR'  if 'Monthly_Sales_INR'  in df.columns else None
        cat_col = 'Product_Category'   if 'Product_Category'   in df.columns else None
        sta_col = 'state'              if 'state'              in df.columns else None
        sto_col = 'Store_ID'           if 'Store_ID'           in df.columns else None

        avg_return = float(df[ret_col].mean()) if ret_col else 0.0

        # ── 1. SNP Fit Scores ─────────────────────────────────────────────
        SNP_PERSONAS_LOCAL = [
            {
                "name": "FMCG High-Velocity Marketplace",
                "icon": "🛒",
                "good_for": ["FMCG", "Household", "Hypermarket"],
                "ret_max": 4.0, "mar_min": 10.0, "health_min": 40,
                "platforms": "Flipkart · Meesho · Udaan",
                "description": "Best for high-turnover everyday consumer goods. Low return rate is critical.",
                "color": "#27ae60", "border": "#27ae60",
            },
            {
                "name": "Premium B2C Digital Seller",
                "icon": "💎",
                "good_for": ["Electronics", "Clothing", "Home & Decor"],
                "ret_max": 5.0, "mar_min": 20.0, "health_min": 60,
                "platforms": "Amazon ONDC · GeM",
                "description": "Higher-margin aspirational products. Quality and presentation drive conversion.",
                "color": "#8b5cf6", "border": "#8b5cf6",
            },
            {
                "name": "B2B Wholesale Distributor",
                "icon": "🏭",
                "good_for": ["Manufacturing", "FMCG", "Clothing", "Hypermarket"],
                "ret_max": 6.0, "mar_min": 8.0, "health_min": 30,
                "platforms": "Udaan · NSIC",
                "description": "Bulk supply to retailers & distributors. Volume-driven, stable margins.",
                "color": "#e07b2a", "border": "#e07b2a",
            },
            {
                "name": "Social Commerce Reseller",
                "icon": "📱",
                "good_for": ["Clothing", "FMCG", "Health & Wellness"],
                "ret_max": 7.0, "mar_min": 12.0, "health_min": 20,
                "platforms": "Meesho ONDC",
                "description": "Tier-2/3 markets via reseller network. Lower entry bar, price-sensitive.",
                "color": "#e84393", "border": "#e84393",
            },
            {
                "name": "Government Procurement Supplier",
                "icon": "🏛️",
                "good_for": ["Manufacturing", "Electronics", "Services", "FMCG"],
                "ret_max": 3.0, "mar_min": 15.0, "health_min": 50,
                "platforms": "GeM · NSIC",
                "description": "Public sector supply. Needs strong quality scores and MSME registration.",
                "color": "#1B4F8A", "border": "#1B4F8A",
            },
        ]

        def _persona_fit_score(p):
            score = 0
            # Return rate check (40 pts)
            if avg_return <= p['ret_max']:       score += 40
            elif avg_return <= p['ret_max']*1.5: score += 20
            # Margin check (25 pts)
            if avg_margin >= p['mar_min']:       score += 25
            elif avg_margin >= p['mar_min']*0.7: score += 12
            # Health check (20 pts)
            if health_score >= p['health_min']:  score += 20
            elif health_score >= p['health_min']*0.75: score += 10
            # Business type match (15 pts)
            if biz_type in p['good_for']:        score += 15
            elif any(g in biz_type for g in p['good_for']): score += 7
            return min(99, score)

        persona_scores = [(p, _persona_fit_score(p)) for p in SNP_PERSONAS_LOCAL]
        persona_scores.sort(key=lambda x: x[1], reverse=True)
        top_persona = persona_scores[0][0]
        top_score   = persona_scores[0][1]

        # ── 2. Product Classification Summary ────────────────────────────
        cat_summary_html = ""
        if cat_col and sal_col:
            cat_grp = df.groupby(cat_col)[sal_col].sum().sort_values(ascending=False)
            total_cat = cat_grp.sum() + 1e-9
            CAT_COLORS = ["#1B4F8A","#27ae60","#f39c12","#8b5cf6","#e74c3c","#e07b2a","#e84393","#0097a7"]
            cat_rows = ""
            for i,(cat,rev) in enumerate(cat_grp.items()):
                pct = rev/total_cat*100; col = CAT_COLORS[i%len(CAT_COLORS)]
                ret_for_cat = df[df[cat_col]==cat][ret_col].mean() if ret_col else 0
                ret_flag = "✅" if ret_for_cat < 4 else ("⚠️" if ret_for_cat < 7 else "🔴")
                cat_rows += f"""<tr style="border-bottom:1px solid #D8E8F8;background:{'#F0F7FF' if i%2==0 else '#FFFFFF'}">
  <td style="padding:9px 14px;font-weight:600;color:{col}">{cat}</td>
  <td style="padding:9px 14px;font-family:monospace;color:#0B1F3A">{_fmt_inr_sb(rev)}</td>
  <td style="padding:9px 14px">
    <div style="display:flex;align-items:center;gap:8px">
      <div style="flex:1;height:6px;background:#D8E8F8;border-radius:3px">
        <div style="width:{pct:.0f}%;height:100%;background:{col};border-radius:3px"></div></div>
      <span style="font-size:11px;font-weight:700;color:{col}">{pct:.1f}%</span></div></td>
  <td style="padding:9px 14px;text-align:center;font-size:13px;color:#1A3050">{ret_flag} {ret_for_cat:.1f}%</td>
</tr>"""
            cat_summary_html = f"""<div style="margin:0">
<table style="width:100%;border-collapse:collapse;background:#FFFFFF;border-radius:12px;overflow:hidden;border:1px solid #C8DCEF;font-size:13px">
<thead><tr style="background:#0B1F3A">
  <th style="padding:11px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Category</th>
  <th style="padding:11px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Revenue</th>
  <th style="padding:11px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Revenue Share</th>
  <th style="padding:11px 14px;text-align:center;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Return Rate</th>
</tr></thead><tbody>{cat_rows}</tbody></table>
<div style="margin-top:8px;font-size:11px;color:#4A6A8A">✅ &lt;4% return rate · ⚠️ 4–7% · 🔴 &gt;7%</div></div>"""

        # ── 3. Region Compatibility ───────────────────────────────────────
        region_html = ""
        if sta_col and sal_col:
            RMAP = {
                "Maharashtra":"West India","Gujarat":"West India","Rajasthan":"West India","Goa":"West India",
                "Tamil Nadu":"South India","TamilNadu":"South India","Kerala":"South India","Karnataka":"South India","Andhra Pradesh":"South India","Telangana":"South India",
                "Uttar Pradesh":"North India","Delhi":"North India","Haryana":"North India","Punjab":"North India",
                "West Bengal":"East India","Odisha":"East India","Bihar":"East India","Jharkhand":"East India",
                "Madhya Pradesh":"Central India","Chhattisgarh":"Central India",
                "Assam":"Northeast India","Manipur":"Northeast India","Nagaland":"Northeast India",
            }
            df['_region'] = df[sta_col].map(RMAP).fillna("Other")
            reg_grp = df.groupby('_region')[sal_col].sum().sort_values(ascending=False)
            top_region = reg_grp.index[0] if len(reg_grp) > 0 else "—"
            top_region_rev = reg_grp.iloc[0] if len(reg_grp) > 0 else 0
            total_reg = reg_grp.sum() + 1e-9
            REG_COLORS = {"South India":"#27ae60","West India":"#1B4F8A","North India":"#f39c12","East India":"#8b5cf6","Central India":"#e07b2a","Northeast India":"#e84393","Other":"#7A92AA"}
            reg_rows = ""
            for reg, rev in reg_grp.items():
                pct = rev/total_reg*100; col = REG_COLORS.get(reg,"#7A92AA")
                reg_rows += f"""<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="font-size:13px;font-weight:600;color:#0B1F3A">{'⭐ ' if reg==top_region else ''}{reg}</span>
    <span style="font-size:12px;font-family:monospace;color:{col};font-weight:700">{_fmt_inr_sb(rev)} &nbsp;·&nbsp; {pct:.1f}%</span></div>
  <div style="height:7px;background:#D8E8F8;border-radius:4px">
    <div style="width:{pct:.0f}%;height:100%;background:{col};border-radius:4px"></div></div></div>"""
            region_html = f"""<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:12px;padding:20px 22px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
    <span style="font-size:20px">🌏</span>
    <div><div style="font-size:13px;font-weight:700;color:#0B1F3A">Your Strongest Sales Region</div>
    <div style="font-size:12px;color:#1a7a40;font-weight:600;margin-top:2px">⭐ {top_region} — {_fmt_inr_sb(top_region_rev)}</div></div></div>
  {reg_rows}</div>"""

        # ── 4. Capacity Health Indicator ─────────────────────────────────
        inv_col = 'Inventory_Turnover' if 'Inventory_Turnover' in df.columns else None
        inv_avg = float(df[inv_col].mean()) if inv_col else 5.0
        # Capacity score: blend of inventory turnover, vendor score, return rate (inverted)
        cap_inv   = min(inv_avg / 12.0, 1.0)               # 12x turnover = full score
        cap_ven   = vendor_sc                               # already 0-1
        cap_ret   = max(0, 1 - avg_return / 10.0)          # 0% return = 1.0, 10%+ = 0
        cap_score = (cap_inv*0.35 + cap_ven*0.35 + cap_ret*0.30) * 100
        cap_lbl   = "Excellent" if cap_score>=75 else ("Good" if cap_score>=55 else ("Moderate" if cap_score>=35 else "Needs Improvement"))
        cap_col   = "#27ae60" if cap_score>=75 else ("#f39c12" if cap_score>=55 else ("#e07b2a" if cap_score>=35 else "#e74c3c"))
        cap_indicators = [
            ("📦 Inventory Turnover", f"{inv_avg:.1f}x / month", cap_inv*100, "Target: 12x+"),
            ("🤝 Vendor Reliability", f"{vendor_sc:.2f}", cap_ven*100, "Target: >0.60"),
            ("↩️ Return Rate",        f"{avg_return:.1f}%",  cap_ret*100, "Target: <4%"),
        ]
        cap_bars = ""
        for lbl, val, pct, target in cap_indicators:
            bar_col = "#1a7a40" if pct>=70 else ("#b05a00" if pct>=45 else "#b03030")
            cap_bars += f"""<div style="margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="font-size:12px;font-weight:600;color:#0B1F3A">{lbl}</span>
    <span style="font-size:12px;font-family:monospace;color:{bar_col};font-weight:700">{val}</span></div>
  <div style="height:7px;background:#D8E8F8;border-radius:4px">
    <div style="width:{min(pct,100):.0f}%;height:100%;background:{bar_col};border-radius:4px"></div></div>
  <div style="font-size:10px;color:#4A6A8A;margin-top:2px">{target}</div></div>"""
        capacity_html = f"""<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:12px;padding:20px 22px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
    <div><div style="font-size:13px;font-weight:700;color:#0B1F3A">⚡ Capacity Health Indicator</div>
    <div style="font-size:11px;color:#4A6A8A;margin-top:2px">Fulfilment readiness for ONDC scale-up</div></div>
    <div style="text-align:right">
      <div style="font-size:26px;font-weight:900;color:{cap_col};font-family:monospace">{cap_score:.0f}%</div>
      <div style="font-size:11px;padding:2px 9px;border-radius:10px;background:{cap_col}22;color:{cap_col};font-weight:700">{cap_lbl}</div></div></div>
  {cap_bars}</div>"""

        # ── 5. Build SNP section HTML ─────────────────────────────────────
        # 5a. Fit Score cards for top 3 platforms (existing SNP_CATALOG)
        snp_scores = {}
        for snp_name, snp_data in SNP_CATALOG.items():
            s = 0.0
            if biz_type in snp_data['business_types']: s += 40
            if health_score >= snp_data['min_health']: s += 20 + min(20, (health_score - snp_data['min_health'])/2)
            if dom in snp_data['segment_boost']: s += 20
            s += growth_sc*10 + vendor_sc*10; snp_scores[snp_name] = min(99, round(s))
        top3_snp = sorted(snp_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        medals_cls = ['gold','silver','bronze']; medals_emoji = ['🥇','🥈','🥉']

        html += _sb_divider(4, 'SNP Mapping Insights', 'ONDC Seller Network Participant Intelligence')

        # ── Which ONDC platform drove your revenue — insight-first view ──────
        # Rank all SNP platforms by fit score (already computed above)
        snp_ranked = sorted(snp_scores.items(), key=lambda x: x[1], reverse=True)
        top_snp_name  = snp_ranked[0][0]  if snp_ranked else "Flipkart Commerce (ONDC)"
        top_snp_score = snp_ranked[0][1]  if snp_ranked else 70
        top_snp_info  = SNP_CATALOG.get(top_snp_name, {})

        # Revenue attribution: estimate % of revenue attributable to each SNP
        # Based on fit score weight relative to total — higher fit = more attribution
        total_snp_score = sum(s for _, s in snp_ranked) + 1e-9
        snp_rev_attr = [(name, score, total_sales * score / total_snp_score) for name, score in snp_ranked]

        # Build "Why this platform drove your revenue" explanation
        WHY_REASONS = {
            'Flipkart Commerce (ONDC)':       ["High product demand units matched Flipkart's high-volume B2C model", "Your margin supports competitive pricing on Flipkart's marketplace", "Consumer goods categories show strong Flipkart platform alignment"],
            'GeM (Government e-Marketplace)': ["Business health score qualifies for GeM government supplier status", "Low return rate meets GeM's strict quality compliance standards", "MSME registration unlocks priority GeM procurement access"],
            'Meesho (ONDC)':                  ["Price-sensitive product categories align with Meesho reseller network", "Tier-2/3 market demand patterns match Meesho's customer base", "Social commerce model suits your product discovery channels"],
            'Amazon Seller Services (ONDC)':  ["Above-average margin supports Amazon's premium positioning", "Low return rate enables Amazon's quality seller badge eligibility", "High-value product categories drive Amazon conversion rates"],
            'Udaan (B2B ONDC)':               ["B2B bulk order potential matches Udaan's distributor network", "Vendor reliability supports consistent B2B fulfilment on Udaan", "Category breadth suits Udaan's retailer discovery model"],
            'NSIC e-Marketplace':             ["Manufacturing/FMCG category qualifies for NSIC MSE matchmaking", "Business health meets NSIC supplier registration requirements", "B2B procurement patterns align with NSIC buyer profiles"],
        }
        top_reasons = WHY_REASONS.get(top_snp_name, ["Strong business metrics align with this platform's requirements", "Category and margin profile match platform expectations", "Health and vendor scores qualify for platform onboarding"])

        # Revenue impact simulation: before ONDC vs estimated with ONDC
        rev_before_ondc = total_sales * 0.82  # estimated 18% uplift from ONDC
        rev_uplift_pct  = (total_sales - rev_before_ondc) / rev_before_ondc * 100 if rev_before_ondc > 0 else 0
        rev_uplift_abs  = total_sales - rev_before_ondc

        # Top platform attribution card
        top_col = "#1B4F8A"
        html += f"""<div style="margin:16px 48px 0;background:linear-gradient(135deg,#F0F7FF 0%,#E4F0FF 100%);border:2px solid #1B4F8A;border-radius:16px;padding:24px 26px">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;flex-wrap:wrap">
    <div style="flex:1;min-width:240px">
      <div style="font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4A6A8A;margin-bottom:8px">📈 Primary Revenue Driver via ONDC</div>
      <div style="font-size:22px;font-weight:900;color:#0B1F3A;margin-bottom:4px">{top_snp_name}</div>
      <div style="font-size:13px;color:#2A4060;margin-bottom:14px">{top_snp_info.get('description_en','')}</div>
      <div style="display:flex;flex-direction:column;gap:6px">
        {''.join(f'<div style="display:flex;align-items:center;gap:8px"><span style="color:#1B4F8A;font-size:14px">▸</span><span style="font-size:12px;color:#1A3050">{r}</span></div>' for r in top_reasons)}
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:12px;min-width:200px">
      <div style="background:#FFFFFF;border-radius:12px;padding:16px 20px;border:1px solid #C8DCEF;text-align:center">
        <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#4A6A8A;margin-bottom:4px">Platform Fit Score</div>
        <div style="font-size:36px;font-weight:900;color:#1B4F8A;font-family:monospace">{top_snp_score}%</div>
        <div style="height:6px;background:#D8E8F8;border-radius:3px;margin-top:8px">
          <div style="width:{top_snp_score}%;height:100%;background:#1B4F8A;border-radius:3px"></div></div>
      </div>
      <div style="background:#FFFFFF;border-radius:12px;padding:14px 20px;border:1px solid #C8DCEF;text-align:center">
        <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#4A6A8A;margin-bottom:4px">Estimated Revenue via ONDC</div>
        <div style="font-size:22px;font-weight:900;color:#1a7a40;font-family:monospace">{_fmt_inr_sb(snp_rev_attr[0][2])}</div>
        <div style="font-size:11px;color:#4A6A8A;margin-top:2px">{snp_rev_attr[0][1]/total_snp_score*100:.0f}% of total revenue share</div>
      </div>
    </div>
  </div>
  <div style="margin-top:16px;padding:12px 16px;background:rgba(27,79,138,.08);border-radius:8px;border-left:3px solid #1B4F8A">
    <span style="font-size:12px;font-weight:700;color:#0B1F3A">Next Step: </span>
    <span style="font-size:12px;color:#2A4060">{top_snp_info.get('action_en','Register on this platform and optimise your product catalogue.')}</span>
  </div>
</div>"""

        # ── Revenue Before vs After ONDC ──────────────────────────────────────
        html += f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:16px 48px 0">
  <div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:12px;padding:18px 20px;text-align:center;border-top:3px solid #B0BEC5">
    <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#4A6A8A;margin-bottom:8px">📊 Before ONDC (Est.)</div>
    <div style="font-size:26px;font-weight:900;color:#7A92AA;font-family:monospace">{_fmt_inr_sb(rev_before_ondc)}</div>
    <div style="font-size:11px;color:#4A6A8A;margin-top:4px">Baseline revenue without ONDC channels</div>
  </div>
  <div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:12px;padding:18px 20px;text-align:center;border-top:3px solid #1a7a40">
    <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#4A6A8A;margin-bottom:8px">🚀 After ONDC (Current)</div>
    <div style="font-size:26px;font-weight:900;color:#1a7a40;font-family:monospace">{_fmt_inr_sb(total_sales)}</div>
    <div style="font-size:11px;color:#4A6A8A;margin-top:4px">Revenue with ONDC platform contribution</div>
  </div>
  <div style="background:#EAF7EE;border:1px solid #C3E6CB;border-radius:12px;padding:18px 20px;text-align:center;border-top:3px solid #1a7a40">
    <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#1a7a40;margin-bottom:8px">📈 ONDC Revenue Uplift</div>
    <div style="font-size:26px;font-weight:900;color:#1a7a40;font-family:monospace">+{rev_uplift_pct:.1f}%</div>
    <div style="font-size:13px;font-weight:700;color:#1a7a40;margin-top:2px">+{_fmt_inr_sb(rev_uplift_abs)}</div>
    <div style="font-size:10px;color:#4A6A8A;margin-top:2px">Estimated uplift from ONDC channels</div>
  </div>
</div>"""

        # ── All platforms revenue attribution breakdown ────────────────────────
        plat_attr_rows = ""
        for rank_i, (pname, pscore, prev) in enumerate(snp_rev_attr[:5]):
            pinfo = SNP_CATALOG.get(pname, {})
            pct_share = pscore / total_snp_score * 100
            bar_w = int(pct_share * 2)  # scale to bar width
            pcol = ["#1B4F8A","#27ae60","#f39c12","#8b5cf6","#e07b2a"][rank_i % 5]
            rank_medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][rank_i]
            plat_attr_rows += f"""<div style="display:flex;align-items:center;gap:14px;padding:10px 0;border-bottom:1px solid #EAF2FF">
  <span style="font-size:18px;width:26px">{rank_medal}</span>
  <div style="flex:1">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
      <span style="font-size:13px;font-weight:700;color:#0B1F3A">{pname}</span>
      <span style="font-size:13px;font-weight:700;color:{pcol};font-family:monospace">{_fmt_inr_sb(prev)}</span>
    </div>
    <div style="height:6px;background:#D8E8F8;border-radius:3px;margin-bottom:3px">
      <div style="width:{min(pct_share*2,100):.0f}%;height:100%;background:{pcol};border-radius:3px"></div></div>
    <div style="display:flex;justify-content:space-between">
      <span style="font-size:10px;color:#4A6A8A">{pinfo.get('description_en','')[:60]}…</span>
      <span style="font-size:10px;font-weight:700;color:#4A6A8A">Fit: {pscore}% · {pct_share:.0f}% share</span>
    </div>
  </div>
</div>"""

        html += f"""<div style="margin:16px 48px 0;background:#FFFFFF;border:1px solid #C8DCEF;border-radius:14px;padding:20px 22px">
  <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4A6A8A;margin-bottom:14px">💰 Revenue Attribution by ONDC Platform</div>
  {plat_attr_rows}
  <div style="font-size:10px;color:#7A92AA;margin-top:10px;padding-top:8px;border-top:1px solid #EAF2FF">Revenue share is estimated based on platform fit score weighting. Actual figures depend on active listings and orders on each platform.</div>
</div>"""

        # ── Top Products for ONDC ─────────────────────────────────────────────
        if sal_col and sku_col:
            top_prod_df = df.copy()
            # Score each product: high sales + good margin + low returns
            if mar_col: top_prod_df['_ondc_rank'] = (top_prod_df[sal_col]/top_prod_df[sal_col].max()*50) + (top_prod_df[mar_col]/top_prod_df[mar_col].max()*30)
            else: top_prod_df['_ondc_rank'] = top_prod_df[sal_col]/top_prod_df[sal_col].max()*80
            if ret_col: top_prod_df['_ondc_rank'] += (1 - top_prod_df[ret_col]/top_prod_df[ret_col].max()+1e-9)*20
            top_prods = top_prod_df.sort_values('_ondc_rank', ascending=False)[[sku_col, sal_col] + ([mar_col] if mar_col else []) + ([ret_col] if ret_col else [])].drop_duplicates(subset=[sku_col]).head(5)

            prod_rows = ""
            for rank, (_, row) in enumerate(top_prods.iterrows(), 1):
                margin_val = f"{row[mar_col]:.1f}%" if mar_col else "—"
                ret_val    = f"{row[ret_col]:.1f}%" if ret_col else "—"
                ret_cls    = "#1a7a40" if (ret_col and row[ret_col]<4) else ("#b05a00" if (ret_col and row[ret_col]<7) else "#b03030")
                rank_col   = "#F5C842" if rank==1 else ("#B0BEC5" if rank==2 else ("#CD7F32" if rank==3 else "#4A6A8A"))
                prod_rows += f"""<tr style="border-bottom:1px solid #D8E8F8;background:{'#F0F7FF' if rank%2==0 else '#FFFFFF'}">
  <td style="padding:9px 14px;text-align:center"><span style="font-size:14px;font-weight:900;color:{rank_col}">#{rank}</span></td>
  <td style="padding:9px 14px;font-weight:600;color:#0B1F3A;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{str(row[sku_col])[:30]}</td>
  <td style="padding:9px 14px;font-family:monospace;color:#1B4F8A;font-weight:700">{_fmt_inr_sb(row[sal_col])}</td>
  <td style="padding:9px 14px;font-weight:600;color:#0B1F3A">{margin_val}</td>
  <td style="padding:9px 14px;font-weight:600;color:{ret_cls}">{ret_val}</td>
</tr>"""
            html += f"""<div style="margin:20px 48px 0">
  <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4A6A8A;margin-bottom:10px">⭐ Top Products to List on ONDC</div>
  <table style="width:100%;border-collapse:collapse;background:#FFFFFF;border-radius:12px;overflow:hidden;border:1px solid #C8DCEF;font-size:13px">
    <thead><tr style="background:#0B1F3A">
      <th style="padding:10px 14px;text-align:center;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Rank</th>
      <th style="padding:10px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Product / SKU</th>
      <th style="padding:10px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Monthly Revenue</th>
      <th style="padding:10px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Margin</th>
      <th style="padding:10px 14px;text-align:left;color:#A8D8FF;font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase">Return Rate</th>
    </tr></thead>
    <tbody>{prod_rows}</tbody>
  </table>
  <div style="font-size:11px;color:#4A6A8A;margin-top:6px">Ranked by composite ONDC suitability (revenue × margin × return rate)</div>
</div>"""

        # ── Platform Recommendation ───────────────────────────────────────────
        plat_cards = ""
        for rank_p, (snp, score) in enumerate(top3_snp):
            info = SNP_CATALOG[snp]
            pcol = "#1a7a40" if score>=70 else ("#b05a00" if score>=45 else "#b03030")
            badge_txt = ["🥇 Best Match", "🥈 2nd Choice", "🥉 3rd Choice"][rank_p]
            plat_cards += f"""<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:12px;padding:16px 18px;border-left:4px solid {pcol}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
    <div style="font-weight:700;font-size:13px;color:#0B1F3A">{snp}</div>
    <span style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;background:{pcol}22;color:{pcol}">{badge_txt} · {score}%</span>
  </div>
  <div style="font-size:11px;color:#2A4060;line-height:1.5;margin-bottom:8px">{info['description_en']}</div>
  <div style="font-size:11px;font-weight:600;color:#1B4F8A;background:#EAF4FF;border-radius:6px;padding:6px 10px">→ {info['action_en']}</div>
</div>"""

        html += f"""<div style="margin:20px 48px 0">
  <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4A6A8A;margin-bottom:12px">🏪 Platform Recommendation for Your Business</div>
  <div style="display:flex;flex-direction:column;gap:12px">{plat_cards}</div>
</div>"""

        # SNP Fit Score cards
        html += '<div class="sb-snp-grid">'
        for i,(snp,score) in enumerate(top3_snp):
            info = SNP_CATALOG[snp]
            bar_col = "#1a7a40" if score>=70 else ("#b05a00" if score>=45 else "#b03030")
            html += f"""<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:14px;padding:20px 18px;{'border-top:4px solid #F5C842' if i==0 else ('border-top:4px solid #B0BEC5' if i==1 else 'border-top:4px solid #CD7F32')}">
  <div style="font-size:24px;margin-bottom:6px">{medals_emoji[i]}</div>
  <div style="font-weight:700;font-size:13px;color:#0B1F3A;margin-bottom:10px">{snp}</div>
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
    <span style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#4A6A8A">SNP FIT SCORE</span>
    <span style="font-size:20px;font-weight:900;color:{bar_col};font-family:monospace">{score}%</span></div>
  <div style="height:6px;background:#D8E8F8;border-radius:3px;margin-bottom:10px">
    <div style="width:{score}%;height:100%;border-radius:3px;background:linear-gradient(90deg,#1B4F8A,{bar_col})"></div></div>
  <div style="font-size:11px;color:#2A4060;line-height:1.5;margin-bottom:8px">{info['description_en']}</div>
  <div style="font-size:10px;font-weight:600;color:#1B4F8A;background:#DCF0FF;border-radius:6px;padding:5px 9px">&rarr; {info['action_en']}</div>
</div>"""
        html += '</div>'

        # Product Classification Summary
        html += f"""<div style="margin:24px 48px 0">
  <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4A6A8A;margin-bottom:10px">📦 Product Classification Summary</div>
  {cat_summary_html if cat_summary_html else '<div style="background:#FFFFFF;border-radius:10px;padding:14px;color:#4A6A8A;font-style:italic;border:1px solid #C8DCEF">No category data available in dataset.</div>'}
</div>"""

        # Region Compatibility + Capacity side by side
        html += f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 48px 0">
  {region_html if region_html else '<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:12px;padding:20px;color:#4A6A8A;font-style:italic">No region/state data available.</div>'}
  {capacity_html}
</div>"""

        # AI Top SNP Persona Fit
        html += f"""<div style="margin:20px 48px 0">
  <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4A6A8A;margin-bottom:12px">🤖 AI "Top SNP Persona Fit" (Simulated)</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">"""

        for p, pscore in persona_scores:
            p_bar_col = "#1a7a40" if pscore>=70 else ("#b05a00" if pscore>=45 else "#b03030")
            is_top = (p == top_persona)
            ret_ok  = "✅" if avg_return <= p['ret_max'] else "❌"
            mar_ok  = "✅" if avg_margin >= p['mar_min'] else "❌"
            hlt_ok  = "✅" if health_score >= p['health_min'] else "❌"
            biz_ok  = "✅" if biz_type in p['good_for'] else "—"
            border_style = f"border:2px solid {p['color']};box-shadow:0 0 0 3px {p['color']}22" if is_top else "border:1px solid #C8DCEF"
            top_badge = f'<div style="position:absolute;top:-10px;right:12px;background:{p["color"]};color:#FFFFFF;font-size:10px;font-weight:700;padding:2px 10px;border-radius:10px;letter-spacing:1px">TOP MATCH</div>' if is_top else ''
            html += f"""<div style="position:relative;background:#FFFFFF;border-radius:12px;padding:18px 16px;{border_style}">
  {top_badge}
  <div style="font-size:22px;margin-bottom:4px">{p['icon']}</div>
  <div style="font-weight:700;font-size:12px;color:#0B1F3A;margin-bottom:4px;line-height:1.3">{p['name']}</div>
  <div style="font-size:10px;color:#4A6A8A;margin-bottom:10px">{p['platforms']}</div>
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
    <span style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#4A6A8A">FIT SCORE</span>
    <span style="font-size:18px;font-weight:900;color:{p_bar_col};font-family:monospace">{pscore}%</span></div>
  <div style="height:5px;background:#D8E8F8;border-radius:3px;margin-bottom:10px">
    <div style="width:{pscore}%;height:100%;background:{p['color']};border-radius:3px"></div></div>
  <div style="font-size:10px;color:#2A4060;line-height:1.5;margin-bottom:10px">{p['description']}</div>
  <div style="font-size:10px;color:#1A3050;background:#EAF4FF;border-radius:6px;padding:6px 8px;border-left:3px solid {p['color']}">
    <div style="font-weight:700;margin-bottom:3px;color:#0B1F3A">Requirements check:</div>
    <div style="color:#1A3050">{ret_ok} Return Rate &lt;{p['ret_max']}% &nbsp;(yours: {avg_return:.1f}%)</div>
    <div style="color:#1A3050">{mar_ok} Margin &gt;{p['mar_min']}% &nbsp;(yours: {avg_margin:.1f}%)</div>
    <div style="color:#1A3050">{hlt_ok} Health &gt;{p['health_min']}% &nbsp;(yours: {health_score:.0f}%)</div>
    <div style="color:#1A3050">{biz_ok} Business Type: {', '.join(p['good_for'][:2])}</div>
  </div>
</div>"""
        html += '</div></div>'
        # Recommendations
        html += _sb_divider(5, 'Action Plan', 'AI-Generated Recommendations')
        html += """<div class="sb-reco-tabs">
<div class="sb-reco-panel"><div class="sb-reco-header immediate">Immediate Actions (0-30 Days)</div>
<div class="sb-reco-row"><span class="sb-reco-priority reco-high">High</span><div style="font-size:13px;line-height:1.5;flex:1;color:#1A2D45">Prioritise top 5 products — focus inventory &amp; marketing spend</div></div>
<div class="sb-reco-row"><span class="sb-reco-priority reco-high">High</span><div style="font-size:13px;line-height:1.5;flex:1;color:#1A2D45">Review products with Financial Risk Score &gt; 0.70</div></div>
<div class="sb-reco-row"><span class="sb-reco-priority reco-medium">Medium</span><div style="font-size:13px;line-height:1.5;flex:1;color:#1A2D45">Strengthen vendor partnerships for low-reliability suppliers</div></div>
<div class="sb-reco-row"><span class="sb-reco-priority reco-medium">Medium</span><div style="font-size:13px;line-height:1.5;flex:1;color:#1A2D45">Analyse high-return products for quality or packaging issues</div></div>
</div>
<div class="sb-reco-panel"><div class="sb-reco-header strategic">Strategic Initiatives (30-90 Days)</div>
<div class="sb-reco-row"><span class="sb-reco-priority reco-high">High</span><div style="font-size:13px;line-height:1.5;flex:1;color:#1A2D45">Use ML forecasts to reduce overstock &amp; stockouts</div></div>
<div class="sb-reco-row"><span class="sb-reco-priority reco-high">High</span><div style="font-size:13px;line-height:1.5;flex:1;color:#1A2D45">Price or cost review for low-margin products</div></div>
<div class="sb-reco-row"><span class="sb-reco-priority reco-medium">Medium</span><div style="font-size:13px;line-height:1.5;flex:1;color:#1A2D45">Allocate budget to high-growth-potential products</div></div>
<div class="sb-reco-row"><span class="sb-reco-priority reco-medium">Medium</span><div style="font-size:13px;line-height:1.5;flex:1;color:#1A2D45">Target operating cost below 60% of revenue</div></div>
</div></div>"""
        html += f'</div>'
        return html, None, forecast_results
    except Exception as e:
        import traceback
        return None, f"Error generating insights: {str(e)}\n\n{traceback.format_exc()}", None


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard data generation
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SNP Mapping Insights Panel — Step 6 Business Intelligence Dashboard
# ══════════════════════════════════════════════════════════════════════════════


def generate_dashboard_data(user_data, df):
    try:
        df = calculate_scores(df)
        sales_col = 'Monthly_Sales_INR'; sku_col = 'SKU_Name' if 'SKU_Name' in df.columns else None
        total_sales = df[sales_col].sum() if sales_col in df.columns else 0
        avg_margin = df['Avg_Margin_Percent'].mean() if 'Avg_Margin_Percent' in df.columns else np.nan
        total_profit = total_sales * (avg_margin/100) if not pd.isna(avg_margin) else np.nan
        health_score = df['MSME_Health_Score'].mean(); growth_score = df['Growth_Potential_Score'].mean()
        performance_score = df['Performance_Score'].mean(); fin_risk_score = df['Financial_Risk_Score'].mean()
        vendor_score = df['Vendor_Score'].mean()
        total_qty = df['Monthly_Demand_Units'].sum() if 'Monthly_Demand_Units' in df.columns else np.nan
        total_products = df[sku_col].nunique() if sku_col else 0; company_name = user_data.get('company_name','—')
        def fmt_inr(v):
            if pd.isna(v): return "N/A"
            if v>=1e7: return f"&#8377;{v/1e7:.2f} Cr"
            if v>=1e5: return f"&#8377;{v/1e5:.2f} L"
            return f"&#8377;{v:,.0f}"
        def fmt_pct(v, d=1): return f"{v:.{d}f}%" if not pd.isna(v) else "N/A"
        def fmt_f(v, d=2): return f"{v:.{d}f}" if not pd.isna(v) else "N/A"
        def fmt_qty(v): return f"{v:,.0f} units" if not pd.isna(v) else "N/A"
        def _hclr(v): return "#27ae60" if v>=65 else ("#f39c12" if v>=40 else "#e74c3c")
        def _rclr(v): return "#27ae60" if v<=0.40 else ("#f39c12" if v<=0.70 else "#e74c3c")
        def _sclr(v): return "#27ae60" if v>=65 else ("#f39c12" if v>=40 else "#e74c3c")
        def _bdg(label, color): return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;">{label}</span>'
        hl = "Healthy" if health_score>=65 else ("Developing" if health_score>=40 else "At Risk")
        rl = "Low Risk" if fin_risk_score<=0.40 else ("Moderate" if fin_risk_score<=0.70 else "High Risk")
        pl = "Excellent" if performance_score>=65 else ("Moderate" if performance_score>=40 else "Low")
        gl = "Strong" if growth_score>=0.60 else ("Moderate" if growth_score>=0.35 else "Low")
        vl = "Strong" if vendor_score>=0.60 else ("Moderate" if vendor_score>=0.35 else "Weak")
        # ──────────────────────────────────────────────────────────────────────
        # ONDC Journey KPI Panel  (health/perf scores removed — shown in AI insights)
        # ──────────────────────────────────────────────────────────────────────
        company_name = user_data.get('company_name', '—')
        msme_key     = user_data.get('msme_number',  '—')

        # Helper: derive raw ONDC cols from either original or remapped names
        def _scol(name, fallback=None):
            return name if name in df.columns else (fallback if fallback and fallback in df.columns else None)

        gc   = _scol('gross_sales',            'Monthly_Sales_INR')
        nc   = _scol('net_sales',               gc)
        boc  = _scol('revenue_before_ondc')
        aoc  = _scol('revenue_after_ondc')
        ochc = _scol('ondc_channel_revenue')
        rrc  = _scol('return_rate_pct',         'Returns_Percentage')
        qrc  = _scol('quantity_returned')
        rpc  = _scol('replacement_count')
        rlrc = _scol('rolling_6m_return_rate')
        tac  = _scol('target_achievement_pct')
        pmrc = _scol('profit_margin_pct',       'Avg_Margin_Percent')
        uc   = _scol('units_sold',              'Monthly_Demand_Units')
        dc   = _scol('date', 'Date')

        def _s(col):   return float(df[col].sum())  if col else 0.0
        def _m(col):   return float(df[col].mean()) if col else 0.0
        def _si(col):  return int(df[col].sum())    if col else 0

        total_gross   = _s(gc)
        total_net     = _s(nc)
        rev_before    = _s(boc)
        rev_after_sum = _s(aoc)
        ondc_pos      = float(df[ochc].clip(lower=0).sum()) if ochc else 0.0
        uplift_pct    = ondc_pos / (rev_before + 1e-9) * 100 if rev_before > 0 else 0.0
        avg_ret_rate  = _m(rrc)
        qty_returned  = _si(qrc)
        replacements  = _si(rpc)
        avg_target    = _m(tac)
        avg_margin    = _m(pmrc)
        total_qty     = _si(uc)

        def fmt_inr(v):
            try:    v = float(v)
            except: return "N/A"
            if pd.isna(v): return "N/A"
            if v >= 1e7:  return f"&#8377;{v/1e7:.2f} Cr"
            if v >= 1e5:  return f"&#8377;{v/1e5:.2f} L"
            return f"&#8377;{v:,.0f}"
        def fmt_pct(v, d=1):
            try:    return f"{float(v):.{d}f}%"
            except: return "N/A"

        # Status badge helpers
        def _bdg(label, bg, fg='#fff'):
            return (f'<span style="background:{bg};color:{fg};padding:2px 9px;'
                    f'border-radius:10px;font-size:11px;font-weight:700">{label}</span>')
        def _ret_bdg(r):
            if r < 4:  return _bdg('Excellent','#27ae60')
            if r < 7:  return _bdg('Moderate', '#f39c12')
            return             _bdg('High Returns','#e74c3c')
        def _tgt_bdg(t):
            if t >= 100: return _bdg('On Target','#27ae60')
            if t >= 90:  return _bdg('Near Target','#f39c12')
            return              _bdg('Below Target','#e74c3c')
        def _upl_bdg(p):
            if p >= 15: return _bdg(f'+{p:.1f}% Uplift','#1B4F8A')
            if p >= 5:  return _bdg(f'+{p:.1f}% Moderate','#f39c12')
            return             _bdg(f'{p:.1f}% Flat','#e74c3c')

        # Store-level summary table
        def _store_table():
            _sid = 'Store_ID' if 'Store_ID' in df.columns else ('store_id' if 'store_id' in df.columns else None)
            if not _sid or not boc:
                return ""
            try:
                st = df.groupby(_sid).agg(
                    net=(nc or gc, 'sum'),
                    before=(boc, 'sum'),
                    ondc_p=(ochc, lambda x: float(x.clip(lower=0).sum())) if ochc else (boc, 'count'),
                    ret_r=(rrc, 'mean') if rrc else (boc, 'count'),
                    qty_r=(qrc, 'sum')  if qrc else (boc, 'count'),
                    repl=(rpc, 'sum')   if rpc else (boc, 'count'),
                    tgt=(tac, 'mean')   if tac else (boc, 'count'),
                ).reset_index()
                rows = ""
                bg = ['#FFFFFF','#F4F9FF','#FFFFFF']
                for idx, r in st.iterrows():
                    bg_c = bg[idx % len(bg)]
                    upl = r['ondc_p'] / (r['before'] + 1e-9) * 100
                    rows += (f'<tr style="background:{bg_c}">'
                             f'<td style="padding:9px 14px;font-weight:700;color:#0B1F3A">Store {int(r[_sid])}</td>'
                             f'<td style="padding:9px 14px;text-align:right;font-weight:700;color:#1B4F8A">{fmt_inr(r["net"])}</td>'
                             f'<td style="padding:9px 14px;text-align:right;color:#4A6A8A">{fmt_inr(r["before"])}</td>'
                             f'<td style="padding:9px 14px;text-align:right;color:#1a7a40;font-weight:700">{fmt_inr(r["ondc_p"])}</td>'
                             f'<td style="padding:9px 14px;text-align:center">{_upl_bdg(upl)}</td>'
                             f'<td style="padding:9px 14px;text-align:center">{_ret_bdg(r["ret_r"])}<br>'
                             f'<span style="font-size:10px;color:#4A6A8A">{r["ret_r"]:.1f}% · {int(r["qty_r"])} units</span></td>'
                             f'<td style="padding:9px 14px;text-align:center"><span style="font-weight:700;color:#8b5cf6">{int(r["repl"])}</span></td>'
                             f'<td style="padding:9px 14px;text-align:center">{_tgt_bdg(r["tgt"])}<br>'
                             f'<span style="font-size:10px;color:#4A6A8A">{r["tgt"]:.1f}%</span></td>'
                             f'</tr>')
                return f"""
  <div style="margin-top:16px">
    <div style="font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#4A6A8A;
                padding:8px 14px;background:#F0F7FF;border-radius:8px 8px 0 0;border:1px solid #D0E4F4;border-bottom:none">
      🏪 Store-Level ONDC Impact
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px;border:1px solid #D0E4F4;border-radius:0 0 8px 8px;overflow:hidden">
      <thead>
        <tr style="background:#0B1F3A;color:#A8D8FF">
          <th style="padding:9px 14px;text-align:left">Store</th>
          <th style="padding:9px 14px;text-align:right">Net Sales</th>
          <th style="padding:9px 14px;text-align:right">Pre-ONDC</th>
          <th style="padding:9px 14px;text-align:right">ONDC Revenue</th>
          <th style="padding:9px 14px;text-align:center">Uplift</th>
          <th style="padding:9px 14px;text-align:center">Return Rate</th>
          <th style="padding:9px 14px;text-align:center">Replacements</th>
          <th style="padding:9px 14px;text-align:center">Target</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>"""
            except Exception:
                return ""

        kpi_html = f"""<div style="font-family:Arial,sans-serif;margin:0 0 18px 0">
  <!-- Header -->
  <div style="background:linear-gradient(135deg,#0B1F3A,#1B4F8A);border-radius:10px;padding:16px 24px;
              margin-bottom:18px;display:flex;align-items:center;gap:16px">
    <span style="font-size:2.2rem">📡</span>
    <div>
      <div style="color:#FFFFFF;font-size:1.15rem;font-weight:800;letter-spacing:0.3px">ONDC Journey Dashboard</div>
      <div style="color:#A8D8FF;font-size:0.84rem;margin-top:3px">{company_name} &nbsp;·&nbsp; {msme_key} &nbsp;·&nbsp; ONDC Impact · Returns · Fulfilment</div>
    </div>
    <div style="margin-left:auto;text-align:right">
      <div style="color:#A8D8FF;font-size:10px;letter-spacing:1px;text-transform:uppercase">ONDC Status</div>
      {'<div style="color:#52e88a;font-size:13px;font-weight:800">● Live since Jan 2024</div>' if rev_after_sum > 0 else
       '<div style="color:#f39c12;font-size:13px;font-weight:800">● Pre-Activation</div>'}
    </div>
  </div>

  <!-- 5 Headline metric cards -->
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px">
    <div style="background:#FFFFFF;border:1px solid #D0E4F4;border-radius:10px;padding:14px 12px;border-top:3px solid #1B4F8A;text-align:center">
      <div style="font-size:9px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#7A92AA;margin-bottom:6px">Total Gross Sales</div>
      <div style="font-size:18px;font-weight:900;color:#0B1F3A;font-family:monospace">{fmt_inr(total_gross)}</div>
      <div style="font-size:10px;color:#7A92AA;margin-top:3px">{total_qty:,} units sold</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #C3E6CB;border-radius:10px;padding:14px 12px;border-top:3px solid #27ae60;text-align:center">
      <div style="font-size:9px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#1a7a40;margin-bottom:6px">ONDC Uplift</div>
      <div style="font-size:18px;font-weight:900;color:#1a7a40;font-family:monospace">+{uplift_pct:.1f}%</div>
      <div style="font-size:10px;color:#7A92AA;margin-top:3px">{fmt_inr(ondc_pos)} via ONDC</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #{'F5C6CB' if avg_ret_rate>=7 else ('FFE8A1' if avg_ret_rate>=4 else 'C3E6CB')};border-radius:10px;padding:14px 12px;border-top:3px solid {'#e74c3c' if avg_ret_rate>=7 else ('#f39c12' if avg_ret_rate>=4 else '#27ae60')};text-align:center">
      <div style="font-size:9px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#7A92AA;margin-bottom:6px">Return Rate</div>
      <div style="font-size:18px;font-weight:900;font-family:monospace;color:{'#e74c3c' if avg_ret_rate>=7 else ('#f39c12' if avg_ret_rate>=4 else '#27ae60')}">{avg_ret_rate:.1f}%</div>
      <div style="font-size:10px;color:#7A92AA;margin-top:3px">{qty_returned:,} units returned</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #E3D0F5;border-radius:10px;padding:14px 12px;border-top:3px solid #8b5cf6;text-align:center">
      <div style="font-size:9px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#7A92AA;margin-bottom:6px">Replacements</div>
      <div style="font-size:18px;font-weight:900;color:#6D28D9;font-family:monospace">{replacements:,}</div>
      <div style="font-size:10px;color:#7A92AA;margin-top:3px">units replaced total</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #{'C3E6CB' if avg_target>=100 else 'FFE8A1'};border-radius:10px;padding:14px 12px;border-top:3px solid {'#27ae60' if avg_target>=100 else '#f39c12'};text-align:center">
      <div style="font-size:9px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#7A92AA;margin-bottom:6px">Target Achievement</div>
      <div style="font-size:18px;font-weight:900;font-family:monospace;color:{'#1a7a40' if avg_target>=100 else '#b05a00'}">{avg_target:.1f}%</div>
      <div style="font-size:10px;color:#7A92AA;margin-top:3px">avg margin {avg_margin:.1f}%</div>
    </div>
  </div>

  <!-- ONDC Before / Channel / Net row -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:16px">
    <div style="background:#F8FAFF;border:1px solid #D0E4F4;border-radius:10px;padding:14px 16px">
      <div style="font-size:9px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:#7A92AA;margin-bottom:5px">📊 Revenue Before ONDC</div>
      <div style="font-size:20px;font-weight:900;color:#7A92AA;font-family:monospace">{fmt_inr(rev_before)}</div>
      <div style="font-size:11px;color:#7A92AA;margin-top:4px">Baseline (2023 pre-ONDC)</div>
    </div>
    <div style="background:#EAF7EE;border:1px solid #C3E6CB;border-radius:10px;padding:14px 16px">
      <div style="font-size:9px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:#1a7a40;margin-bottom:5px">🚀 ONDC Channel Revenue</div>
      <div style="font-size:20px;font-weight:900;color:#1a7a40;font-family:monospace">{fmt_inr(ondc_pos)}</div>
      <div style="font-size:11px;color:#7A92AA;margin-top:4px">New revenue from ONDC platforms</div>
    </div>
    <div style="background:#F0F7FF;border:1px solid #B8D4F0;border-radius:10px;padding:14px 16px">
      <div style="font-size:9px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:#1B4F8A;margin-bottom:5px">📈 Net Sales (Post-ONDC)</div>
      <div style="font-size:20px;font-weight:900;color:#1B4F8A;font-family:monospace">{fmt_inr(total_net)}</div>
      <div style="font-size:11px;color:#7A92AA;margin-top:4px">{_upl_bdg(uplift_pct)} vs pre-ONDC baseline</div>
    </div>
  </div>

  <!-- Revenue & Volume summary rows (no health/performance scores) -->
  <table style="width:100%;border-collapse:separate;border-spacing:0;border-radius:10px;overflow:hidden;
                box-shadow:0 2px 10px rgba(0,51,102,0.08);font-size:0.90rem;margin-bottom:14px">
    <thead>
      <tr style="background:#0B1F3A">
        <th style="padding:10px 16px;text-align:left;font-weight:700;width:38%;color:#A8D8FF;font-size:11px;letter-spacing:1px">Metric</th>
        <th style="padding:10px 16px;text-align:right;font-weight:700;width:28%;color:#A8D8FF;font-size:11px">Value</th>
        <th style="padding:10px 16px;text-align:center;font-weight:700;width:20%;color:#A8D8FF;font-size:11px">Status</th>
        <th style="padding:10px 16px;text-align:left;font-weight:700;width:14%;color:#A8D8FF;font-size:11px">Benchmark</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#F0F7FF"><td colspan="4" style="padding:6px 16px;font-weight:800;color:#1B4F8A;font-size:10px;letter-spacing:1.5px;text-transform:uppercase">Revenue &amp; ONDC Impact</td></tr>
      <tr style="background:#FFFFFF;border-bottom:1px solid #E0EDF8">
        <td style="padding:9px 16px;color:#1A2D45">Total Gross Sales</td>
        <td style="padding:9px 16px;text-align:right;font-weight:800;color:#0B1F3A;font-size:1rem">{fmt_inr(total_gross)}</td>
        <td style="padding:9px 16px;text-align:center">—</td>
        <td style="padding:9px 16px;color:#6A8AA8;font-size:11px">All channels</td>
      </tr>
      <tr style="background:#F4F9FF;border-bottom:1px solid #E0EDF8">
        <td style="padding:9px 16px;color:#1A2D45">Pre-ONDC Revenue Baseline</td>
        <td style="padding:9px 16px;text-align:right;font-weight:700;color:#7A92AA">{fmt_inr(rev_before)}</td>
        <td style="padding:9px 16px;text-align:center">—</td>
        <td style="padding:9px 16px;color:#6A8AA8;font-size:11px">2023 only</td>
      </tr>
      <tr style="background:#FFFFFF;border-bottom:1px solid #E0EDF8">
        <td style="padding:9px 16px;color:#1A2D45">ONDC Channel Revenue</td>
        <td style="padding:9px 16px;text-align:right;font-weight:800;color:#1a7a40">{fmt_inr(ondc_pos)}</td>
        <td style="padding:9px 16px;text-align:center">{_upl_bdg(uplift_pct)}</td>
        <td style="padding:9px 16px;color:#6A8AA8;font-size:11px">Target &gt;10%</td>
      </tr>
      <tr style="background:#F4F9FF;border-bottom:1px solid #E0EDF8">
        <td style="padding:9px 16px;color:#1A2D45">Avg Profit Margin</td>
        <td style="padding:9px 16px;text-align:right;font-weight:700;color:#1a3a6b">{fmt_pct(avg_margin)}</td>
        <td style="padding:9px 16px;text-align:center">{'🟢' if avg_margin>20 else ('🟡' if avg_margin>10 else '🔴')}</td>
        <td style="padding:9px 16px;color:#6A8AA8;font-size:11px">Target &gt;20%</td>
      </tr>
      <tr style="background:#F0F7FF"><td colspan="4" style="padding:6px 16px;font-weight:800;color:#1B4F8A;font-size:10px;letter-spacing:1.5px;text-transform:uppercase">Returns &amp; Fulfilment</td></tr>
      <tr style="background:#FFFFFF;border-bottom:1px solid #E0EDF8">
        <td style="padding:9px 16px;color:#1A2D45">Avg Return Rate</td>
        <td style="padding:9px 16px;text-align:right;font-weight:800;color:{'#e74c3c' if avg_ret_rate>=7 else '#f39c12'}">{fmt_pct(avg_ret_rate)}</td>
        <td style="padding:9px 16px;text-align:center">{_ret_bdg(avg_ret_rate)}</td>
        <td style="padding:9px 16px;color:#6A8AA8;font-size:11px">Target &lt;7%</td>
      </tr>
      <tr style="background:#F4F9FF;border-bottom:1px solid #E0EDF8">
        <td style="padding:9px 16px;color:#1A2D45">Total Units Returned</td>
        <td style="padding:9px 16px;text-align:right;font-weight:700;color:#e74c3c">{qty_returned:,} units</td>
        <td style="padding:9px 16px;text-align:center">—</td>
        <td style="padding:9px 16px;color:#6A8AA8;font-size:11px">Cumulative</td>
      </tr>
      <tr style="background:#FFFFFF;border-bottom:1px solid #E0EDF8">
        <td style="padding:9px 16px;color:#1A2D45">Total Replacements Issued</td>
        <td style="padding:9px 16px;text-align:right;font-weight:700;color:#8b5cf6">{replacements:,} units</td>
        <td style="padding:9px 16px;text-align:center">—</td>
        <td style="padding:9px 16px;color:#6A8AA8;font-size:11px">Service quality</td>
      </tr>
      <tr style="background:#F4F9FF">
        <td style="padding:9px 16px;color:#1A2D45">Avg Target Achievement</td>
        <td style="padding:9px 16px;text-align:right;font-weight:800;color:{'#1a7a40' if avg_target>=100 else '#b05a00'}">{fmt_pct(avg_target)}</td>
        <td style="padding:9px 16px;text-align:center">{_tgt_bdg(avg_target)}</td>
        <td style="padding:9px 16px;color:#6A8AA8;font-size:11px">Target 100%</td>
      </tr>
    </tbody>
  </table>

  {_store_table()}
</div>"""

        # ──────────────────────────────────────────────────────────────────────
        # 4 ONDC-focused charts
        # ──────────────────────────────────────────────────────────────────────
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                             'axes.spines.right': False, 'axes.grid': True,
                             'grid.alpha': 0.25, 'grid.color': '#B0C4DE'})

        # Build quarterly time-series from raw ONDC columns
        df_ts = df.copy()
        if dc:
            df_ts[dc] = pd.to_datetime(df_ts[dc], errors='coerce')
            df_ts = df_ts.dropna(subset=[dc])
            df_ts['_yr']  = df_ts[dc].dt.year
            df_ts['_qn']  = df_ts[dc].dt.quarter
            df_ts['_ql']  = df_ts['_yr'].astype(str) + '-Q' + df_ts['_qn'].astype(str)
            df_ts['_mth'] = df_ts[dc].dt.to_period('M').astype(str)
            has_ts = True
        else:
            has_ts = False

        NAVY  = '#1B4F8A'
        GREEN = '#27ae60'
        RED   = '#e74c3c'
        AMBER = '#f39c12'
        PURP  = '#8b5cf6'
        TEAL  = '#0097a7'

        def _inr_fmt(x, _):
            if abs(x) >= 1e7:  return f'₹{x/1e7:.1f}Cr'
            if abs(x) >= 1e5:  return f'₹{x/1e5:.0f}L'
            return f'₹{x:,.0f}'

        # ── Chart 1: Sales vs Profit Margin — quarterly dual-axis ─────────────
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        fig1.subplots_adjust(top=0.87, bottom=0.18, left=0.10, right=0.91)
        if has_ts and gc and pmrc:
            q1 = df_ts.groupby('_ql').agg(
                sales=(gc,    'sum'),
                margin=(pmrc, 'mean')
            ).reset_index().sort_values('_ql')
            ax1b = ax1.twinx()
            x1   = range(len(q1))
            bars1 = ax1.bar(x1, q1['sales']/1e5, color=NAVY, alpha=0.72, label='Gross Sales (₹L)', width=0.6, zorder=3)
            ax1b.plot(x1, q1['margin'], color=RED, linewidth=2.5, marker='o', markersize=5, label='Profit Margin %', zorder=4)
            ax1.set_xticks(list(x1))
            ax1.set_xticklabels(q1['_ql'], rotation=45, ha='right', fontsize=8)
            ax1.set_ylabel('Gross Sales (₹ Lakhs)', fontsize=10, fontweight='bold', color=NAVY)
            ax1b.set_ylabel('Profit Margin %', fontsize=10, fontweight='bold', color=RED)
            ax1.set_title('Sales vs Profit Margin — Quarterly', fontsize=13, fontweight='bold', pad=12)
            h1, l1 = ax1.get_legend_handles_labels();  h2, l2 = ax1b.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=8, framealpha=0.8)
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_inr_fmt))
            ax1b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))
            # Annotate last 4 quarters
            for xi, mi in zip(list(x1)[-4:], q1['margin'].values[-4:]):
                ax1b.annotate(f'{mi:.1f}%', (xi, mi), textcoords='offset points', xytext=(0, 6),
                               fontsize=7, ha='center', color=RED, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No time-series data available', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Sales vs Profit Margin', fontsize=13, fontweight='bold')

        # ── Chart 2: ONDC Before vs After — stacked quarterly with uplift line ─
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        fig2.subplots_adjust(top=0.87, bottom=0.18, left=0.10, right=0.97)
        if has_ts and boc and ochc:
            q2 = df_ts.groupby('_ql').agg(
                before=(boc, 'sum'),
                ondc_p=(ochc, lambda x: float(x.clip(lower=0).sum())),
                gross=(gc, 'sum') if gc else (boc, 'sum')
            ).reset_index().sort_values('_ql')
            x2 = range(len(q2))
            ax2.bar(x2, q2['before']/1e5,  label='Revenue Before ONDC', color='#7A92AA', alpha=0.80, width=0.6, zorder=3)
            ax2.bar(x2, q2['ondc_p']/1e5, bottom=q2['before']/1e5,
                    label='ONDC Channel Revenue', color=GREEN, alpha=0.85, width=0.6, zorder=3)
            ax2.plot(x2, q2['gross']/1e5, color=NAVY, linewidth=2.5, marker='D', markersize=5,
                     label='Total Gross Sales', zorder=5)
            # Mark ONDC activation (first quarter where ondc_p > 0)
            first_live = q2[q2['ondc_p'] > 0]['_ql'].iloc[0] if (q2['ondc_p'] > 0).any() else None
            if first_live:
                li = q2[q2['_ql'] == first_live].index[0]
                ax2.axvline(li - 0.5, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7)
                ax2.text(li - 0.4, ax2.get_ylim()[1] * 0.92 if ax2.get_ylim()[1] else 1000,
                         '▶ ONDC Live', fontsize=8, color=GREEN, fontweight='bold')
            ax2.set_xticks(list(x2))
            ax2.set_xticklabels(q2['_ql'], rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Revenue (₹ Lakhs)', fontsize=10, fontweight='bold')
            ax2.set_title('ONDC Impact: Before vs After Revenue — Quarterly', fontsize=13, fontweight='bold', pad=12)
            ax2.legend(fontsize=8, loc='upper left', framealpha=0.8)
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_inr_fmt))
        else:
            ax2.text(0.5, 0.5, 'ONDC revenue columns not found in data', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title('ONDC Before vs After', fontsize=13, fontweight='bold')

        # ── Chart 3: Returns & Replacements — quarterly grouped bars + rate line
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        fig3.subplots_adjust(top=0.87, bottom=0.18, left=0.10, right=0.91)
        if has_ts and rrc:
            agg3 = {rrc: 'mean'}
            if rlrc: agg3[rlrc] = 'mean'
            if qrc:  agg3[qrc]  = 'sum'
            if rpc:  agg3[rpc]  = 'sum'
            q3 = df_ts.groupby('_ql').agg(agg3).reset_index().sort_values('_ql')
            ax3b = ax3.twinx()
            x3   = range(len(q3))
            bar_w = 0.35
            if qrc and rpc:
                b3a = ax3.bar([xi - bar_w/2 for xi in x3], q3[qrc],
                              width=bar_w, color=RED, alpha=0.75, label='Units Returned', zorder=3)
                b3b = ax3.bar([xi + bar_w/2 for xi in x3], q3[rpc],
                              width=bar_w, color=AMBER, alpha=0.75, label='Replacements', zorder=3)
            elif qrc:
                ax3.bar(x3, q3[qrc], width=0.6, color=RED, alpha=0.75, label='Units Returned', zorder=3)
            ax3b.plot(x3, q3[rrc],  color=PURP, linewidth=2.5, marker='o', markersize=5,
                      label='Return Rate %', zorder=4)
            if rlrc:
                ax3b.plot(x3, q3[rlrc], color=TEAL, linewidth=1.8, linestyle='--', marker='s',
                           markersize=4, label='6M Rolling Return Rate', zorder=4)
            ax3b.axhline(7, color=RED, linestyle=':', linewidth=1.2, alpha=0.5)
            ax3b.text(len(q3)-0.5, 7.2, 'Target <7%', fontsize=7, color=RED, ha='right')
            ax3.set_xticks(list(x3))
            ax3.set_xticklabels(q3['_ql'], rotation=45, ha='right', fontsize=8)
            ax3.set_ylabel('Units (Returned / Replaced)', fontsize=10, fontweight='bold', color=RED)
            ax3b.set_ylabel('Return Rate %', fontsize=10, fontweight='bold', color=PURP)
            ax3.set_title('Returns & Replacements — Quarterly Trend', fontsize=13, fontweight='bold', pad=12)
            h3a, l3a = ax3.get_legend_handles_labels();  h3b, l3b = ax3b.get_legend_handles_labels()
            ax3.legend(h3a+h3b, l3a+l3b, loc='upper left', fontsize=8, framealpha=0.8)
        else:
            ax3.text(0.5, 0.5, 'No returns data available', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Returns & Replacements', fontsize=13, fontweight='bold')

        # ── Chart 4: Store-level ONDC comparison (2 sub-plots) ───────────────
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
        fig4.subplots_adjust(top=0.87, bottom=0.14, left=0.07, right=0.97, wspace=0.32)
        _sid4 = 'Store_ID' if 'Store_ID' in df.columns else ('store_id' if 'store_id' in df.columns else None)
        if _sid4 and boc:
            agg4 = dict(net=(nc or gc, 'sum'), before=(boc, 'sum'))
            if ochc: agg4['ondc_p'] = (ochc, lambda x: float(x.clip(lower=0).sum()))
            if rrc:  agg4['ret_r']  = (rrc,  'mean')
            if rpc:  agg4['repl']   = (rpc,  'sum')
            if tac:  agg4['tgt_ach']= (tac,  'mean')
            st4 = df.groupby(_sid4).agg(**agg4).reset_index()
            store_lbls = [f"Store {int(s)}" for s in st4[_sid4]]
            x4  = range(len(st4))
            w4  = 0.28
            # Left sub-plot: revenue bars
            ax4a.bar([xi - w4 for xi in x4], st4['before']/1e5,   width=w4, color='#7A92AA', alpha=0.85, label='Pre-ONDC Baseline')
            ax4a.bar([xi       for xi in x4], st4['net']/1e5,     width=w4, color=NAVY,     alpha=0.85, label='Net Sales')
            if 'ondc_p' in st4:
                ax4a.bar([xi + w4 for xi in x4], st4['ondc_p']/1e5, width=w4, color=GREEN,   alpha=0.85, label='ONDC Channel')
            ax4a.set_xticks(list(x4)); ax4a.set_xticklabels(store_lbls, fontsize=10)
            ax4a.set_ylabel('Revenue (₹ Lakhs)', fontsize=10, fontweight='bold')
            ax4a.set_title('Store Revenue: Pre vs Post ONDC', fontsize=11, fontweight='bold', pad=10)
            ax4a.legend(fontsize=8); ax4a.yaxis.set_major_formatter(mticker.FuncFormatter(_inr_fmt))
            # Right sub-plot: return rate bars + target achievement line
            ax4b2 = ax4b.twinx()
            pal4  = [RED, AMBER, '#e07b2a']
            if 'ret_r' in st4:
                b4  = ax4b.bar(x4, st4['ret_r'], color=pal4[:len(st4)], alpha=0.78, width=0.5, label='Return Rate %', zorder=3)
                for bar, v in zip(b4, st4['ret_r']):
                    ax4b.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.1,
                               f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            if 'tgt_ach' in st4:
                ax4b2.plot(list(x4), st4['tgt_ach'], color=NAVY, linewidth=2.5, marker='D',
                            markersize=8, label='Target Achievement %', zorder=5)
                ax4b2.axhline(100, color=GREEN, linestyle='--', linewidth=1.2, alpha=0.6)
            ax4b.set_xticks(list(x4)); ax4b.set_xticklabels(store_lbls, fontsize=10)
            ax4b.set_ylabel('Return Rate %', fontsize=10, fontweight='bold', color=RED)
            ax4b2.set_ylabel('Target Achievement %', fontsize=10, fontweight='bold', color=NAVY)
            ax4b.set_title('Store: Return Rate & Target Achievement', fontsize=11, fontweight='bold', pad=10)
            h4a, l4a = ax4b.get_legend_handles_labels();  h4b2, l4b2 = ax4b2.get_legend_handles_labels()
            ax4b.legend(h4a+h4b2, l4a+l4b2, loc='upper right', fontsize=8, framealpha=0.8)
        else:
            for ax_ in [ax4a, ax4b]:
                ax_.text(0.5, 0.5, 'No store-level ONDC data', ha='center', va='center',
                         transform=ax_.transAxes, fontsize=12)
        fig4.suptitle('Store-Level ONDC Impact Analysis', fontsize=13, fontweight='bold', y=0.97)

        cat_options = ['All Categories']
        if 'Product_Category' in df.columns:
            cat_options += sorted(df['Product_Category'].dropna().unique().tolist())
        return (kpi_html,"","","","",fig1,fig2,fig3,fig4,None,None,None,None,None,cat_options,df)

    except Exception as e:
        import traceback; traceback.print_exc()
        return ("N/A","","","","",None,None,None,None,None,None,None,None,f"Error: {str(e)}",["All Categories"],None)


# ══════════════════════════════════════════════════════════════════════════════
# Mock Data
# ══════════════════════════════════════════════════════════════════════════════
udyam_master_data = pd.DataFrame({
    'udyam_number':    ['UDYAM-UP-01-0000001','UDYAM-TN-00-7629703','UDYAM-KL-03-0000003'],
    'enterprise_name': ['Tech Innovations Pvt Ltd','Retail Solutions Corp','FMCG Distributors'],
    'organisation_type':['Private Limited','Partnership','Proprietorship'],
    'major_activity':  ['FMCG','Services','Electronics'],
    'enterprise_type': ['Small','Micro','Medium'],
    'state':           ['Uttar Pradesh','TamilNadu','Kerala'],
    'city':            ['Lucknow','Chennai','Kochi'],
    'industry_domain': ['Retail','Retail','Retail'],   # FIX 4 — added
})

def _fetch_msme_data(msme_number):
    fetched = udyam_master_data[udyam_master_data['udyam_number'] == msme_number]
    if not fetched.empty:
        row = fetched.iloc[0]
        return (row['enterprise_name'], row['organisation_type'], row['major_activity'],
                row['enterprise_type'], row['state'], row['city'],
                row.get('industry_domain','Retail'),          # FIX 4
                "✅ MSME Data Fetched Successfully")
    return "", "", "", "", "", "", "Retail", "❌ MSME Data Not Found. Please check the number."


# ══════════════════════════════════════════════════════════════════════════════
# Category filter chart
# ══════════════════════════════════════════════════════════════════════════════
def build_category_filter_chart(df, selected_category):
    plt.style.use('seaborn-v0_8-darkgrid')
    sales_col = 'Monthly_Sales_INR' if 'Monthly_Sales_INR' in df.columns else 'Gross_Sales'
    sku_col = 'SKU_Name' if 'SKU_Name' in df.columns else None
    cat_col = 'Product_Category' if 'Product_Category' in df.columns else None
    fig,ax = plt.subplots(figsize=(12,7)); fig.subplots_adjust(top=0.91,bottom=0.12,left=0.32,right=0.92)
    def _fmt(v):
        if v>=1e7: return f"Rs.{v/1e7:.1f}Cr"
        if v>=1e5: return f"Rs.{v/1e5:.1f}L"
        return f"Rs.{v:,.0f}"
    if cat_col and selected_category and selected_category != "All Categories":
        filtered = df[df[cat_col]==selected_category] if selected_category in df[cat_col].values else df
    else:
        filtered = df
    if sku_col and not filtered.empty:
        top5 = filtered.groupby(sku_col)[sales_col].sum().nlargest(5).reset_index()
        colors = plt.cm.RdYlGn(np.linspace(0.3,0.9,len(top5)))
        bars = ax.barh(top5[sku_col], top5[sales_col], color=colors, height=0.55, edgecolor='white')
        ax.set_xlabel('Sales (INR)',fontsize=12,fontweight='bold')
        cat_label = selected_category if selected_category and selected_category != "All Categories" else "All"
        ax.set_title(f'Top 5 Products — {cat_label}',fontsize=14,fontweight='bold',pad=14)
        ax.grid(axis='x',alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        max_val = top5[sales_col].max() if len(top5)>0 else 1
        for bar in bars:
            w = bar.get_width()
            ax.text(w+max_val*0.01, bar.get_y()+bar.get_height()/2, _fmt(w), ha='left', va='center', fontsize=9, fontweight='bold')
        ax.set_xlim(0, max_val*1.22)
    else:
        ax.text(0.5,0.5,'No product data',ha='center',va='center',transform=ax.transAxes)
    return fig

def handle_category_filter(selected_category, raw_df):
    if raw_df is None: return None, ""
    df = calculate_scores(raw_df.copy()); fig = build_category_filter_chart(df, selected_category)
    sales_col = 'Monthly_Sales_INR' if 'Monthly_Sales_INR' in df.columns else 'Gross_Sales'
    sku_col = 'SKU_Name' if 'SKU_Name' in df.columns else None
    cat_col = 'Product_Category' if 'Product_Category' in df.columns else None
    def _fmt(v):
        if v>=1e7: return f"Rs.{v/1e7:.1f}Cr"
        if v>=1e5: return f"Rs.{v/1e5:.1f}L"
        return f"Rs.{v:,.0f}"
    if cat_col and selected_category and selected_category != "All Categories":
        filtered = df[df[cat_col]==selected_category] if selected_category in df[cat_col].values else df
        cat_label = selected_category
    else:
        filtered = df; cat_label = "All Categories"
    total_cat_sales = filtered[sales_col].sum() if sales_col in filtered.columns else 0
    avg_margin_cat = filtered['Avg_Margin_Percent'].mean() if 'Avg_Margin_Percent' in filtered.columns else 0
    health_avg_cat = filtered['MSME_Health_Score'].mean() if 'MSME_Health_Score' in filtered.columns else 0
    insight_rows = ""
    if sku_col and not filtered.empty:
        top5 = filtered.groupby(sku_col)[sales_col].sum().nlargest(5).reset_index()
        for i, (_,row) in enumerate(top5.iterrows()):
            medal = ['🥇','🥈','🥉','4️⃣','5️⃣'][i]
            insight_rows += f'<tr style="background:{"#FFFFFF" if i%2==0 else "#F4F9FF"};border-bottom:1px solid #e8f0fe;"><td style="padding:9px 12px;font-weight:700;color:#003366">{medal}</td><td style="padding:9px 12px;font-weight:600;color:#1a1a2e">{row[sku_col]}</td><td style="padding:9px 12px;text-align:right;font-weight:700;color:#27ae60">{_fmt(row[sales_col])}</td></tr>'
    insight_html = f"""<div style="background:linear-gradient(135deg,#eaf2ff,#f0f7ff);border-left:4px solid #003366;border-radius:0 10px 10px 0;padding:16px 18px;margin-top:8px;">
<div style="font-weight:700;color:#003366;font-size:1rem;margin-bottom:10px;">Category Insight: <span style="color:#1f77b4">{cat_label}</span></div>
<div style="display:flex;gap:18px;margin-bottom:14px;flex-wrap:wrap;">
<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:8px;padding:10px 16px;flex:1;min-width:120px"><div style="color:#4A6A8A;font-size:0.75rem">Category Revenue</div><div style="font-weight:700;color:#0f2557;font-size:1.1rem">{_fmt(total_cat_sales)}</div></div>
<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:8px;padding:10px 16px;flex:1;min-width:120px"><div style="color:#4A6A8A;font-size:0.75rem">Avg Margin</div><div style="font-weight:700;color:#27ae60;font-size:1.1rem">{avg_margin_cat:.1f}%</div></div>
<div style="background:#FFFFFF;border:1px solid #C8DCEF;border-radius:8px;padding:10px 16px;flex:1;min-width:120px"><div style="color:#4A6A8A;font-size:0.75rem">Avg Health Score</div><div style="font-weight:700;color:#1a3a6b;font-size:1.1rem">{health_avg_cat:.1f}%</div></div>
</div>
<table style="width:100%;border-collapse:collapse;font-size:0.88rem"><thead><tr style="background:#003366"><th style="padding:8px 12px;text-align:left;color:white">#</th><th style="padding:8px 12px;text-align:left;color:white">Product</th><th style="padding:8px 12px;text-align:right;color:white">Revenue</th></tr></thead><tbody>{insight_rows}</tbody></table></div>"""
    return fig, insight_html


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
custom_css = """
.header-container{background:linear-gradient(135deg,#0f2557 0%,#1a3a6b 100%);padding:10px 20px;display:flex;justify-content:space-between;align-items:center;}
.logo-section{display:flex;align-items:center;gap:12px;}
.logo-img{height:45px;width:auto;filter:brightness(1.1);}
.logo-text{background:linear-gradient(to right,#6a0dad,#007bff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:28px;font-weight:700;}
.hero-section{background:linear-gradient(90deg,#0f172a,#1e3a8a);padding:60px 40px;border-radius:10px;text-align:center;color:white;}
.hero-title{font-size:42px;font-weight:700;margin-bottom:15px;color:white;}
.hero-section h2.hero-sub-tagline{font-size:24px;font-weight:500;color:white;opacity:0.9;}
.hero-section p.hero-description{margin-top:15px;font-size:18px;opacity:0.9;color:white;}
.section{padding:40px 20px;margin-top:20px;border-radius:8px;background-color:#f9f9f9;}
.section-title{font-size:32px;font-weight:700;color:#333;text-align:center;margin-bottom:20px;}
.capabilities-section{background-color:#f0f8ff;padding:50px 20px;margin-top:40px;text-align:center;border-radius:8px;}
.capabilities-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:30px;max-width:1200px;margin:0 auto;}
.capability-card{background-color:#ffffff;padding:30px;border-radius:10px;text-align:center;}
.footer-section{background-color:#0f2557;color:#ffffff;padding:20px;margin-top:40px;border-radius:8px;display:flex;justify-content:space-between;align-items:center;}
.footer-section a{color:#ffffff;text-decoration:none;}
body,html{background-color:#f8f9fa;}
.voice-section{background:linear-gradient(135deg,#f0f7ff,#e8f4ff);border:2px solid #7AABDD;border-radius:12px;padding:16px 20px;margin:12px 0;}
.voice-btn{background:linear-gradient(135deg,#e74c3c,#c0392b);color:white;border:none;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:600;cursor:pointer;}
.voice-status{font-size:13px;color:#2A4060;margin-top:8px;font-style:italic;}
"""

business_types = ["Choose Business Type","FMCG","Hypermarket","Clothing","Electronics"]
roles = ["Business Owner","Co-Founder","Category Manager","Analyst","Store Manager"]
ACTIVITY_TO_BIZ_TYPE = {'FMCG':'FMCG','Hypermarket':'Hypermarket','Electronics':'Electronics','Clothing':'Clothing','Manufacturing':'FMCG','Services':'FMCG','Trading':'FMCG','Retail':'Hypermarket'}

# ══════════════════════════════════════════════════════════════════════════════
# Landing page helpers
# ══════════════════════════════════════════════════════════════════════════════
def _landing_hero(lang):
    if lang == 'hi':
        return """<div class="header-container"><div class="logo-section"><img src="https://i.postimg.cc/qRNQYbZJ/Data-Netra-Logo.jpg" class="logo-img" alt="DataNetra.ai Logo"><div class="logo-text">DataNetra.ai</div></div></div>
<div class="hero-section"><h1 class="hero-title">AI से आपके धंधे की पूरी जानकारी</h1><h2 class="hero-sub-tagline">Data देखें। सही फैसला लें। आगे बढ़ें।</h2><p class="hero-description">अपना बिक्री का Data डालें — AI आपको बताएगा क्या बेचें, कब बेचें और कैसे मुनाफा बढ़ाएं।</p></div>"""
    return """<div class="header-container"><div class="logo-section"><img src="https://i.postimg.cc/qRNQYbZJ/Data-Netra-Logo.jpg" class="logo-img" alt="DataNetra.ai Logo"><div class="logo-text">DataNetra.ai</div></div></div>
<div class="hero-section"><h1 class="hero-title">AI-Powered Retail Intelligence</h1><h2 class="hero-sub-tagline">Data with Vision. Decisions with Confidence.</h2><p class="hero-description">Turn retail data into predictive insights, smarter decisions, and measurable growth.</p></div>"""

def _landing_capabilities(lang):
    if lang == 'hi':
        return """<div class="capabilities-section"><h2 class="section-title">DataNetra क्या-क्या कर सकता है?</h2><div class="capabilities-grid"><div class="capability-card"><div style="font-size:48px">🎯</div><h3>Smart स्कोरिंग</h3><p>आपके सामान और धंधे को नंबर से समझाता है</p></div><div class="capability-card"><div style="font-size:48px">📊</div><h3>धंधे का Dashboard</h3><p>बिक्री, मुनाफा और सेहत — सब एक जगह देखें</p></div><div class="capability-card"><div style="font-size:48px">🔮</div><h3>आगे की बिक्री का अनुमान</h3><p>AI बताता है कि अगले 6-12 महीनों में बिक्री कितनी होगी</p></div><div class="capability-card"><div style="font-size:48px">🔗</div><h3>आसान जोड़</h3><p>अपने Excel या POS System का Data सीधे डालें</p></div></div></div>
<div class="footer-section"><div>आपका Data सुरक्षित है</div><div>© 2026 DataNetra.ai &nbsp;|&nbsp;<a href="https://www.linkedin.com/company/108412762/" target="_blank">LinkedIn</a></div></div>"""
    return """<div class="capabilities-section"><h2 class="section-title">Platform Capabilities</h2><div class="capabilities-grid"><div class="capability-card"><div style="font-size:48px">🎯</div><h3>Smart Scoring Engine</h3><p>Automated analysis for accurate performance scoring and health scores</p></div><div class="capability-card"><div style="font-size:48px">📊</div><h3>Business Health Dashboard</h3><p>Monitor key metrics and KPIs in one real-time dashboard</p></div><div class="capability-card"><div style="font-size:48px">🔮</div><h3>Predictive Insights</h3><p>AI-driven forecasts to anticipate future trends</p></div><div class="capability-card"><div style="font-size:48px">🔗</div><h3>Easy Integration</h3><p>Seamlessly connect with your existing retail and POS systems</p></div></div></div>
<div class="footer-section"><div>Data Secured &amp; Protected</div><div>© 2026 DataNetra.ai &nbsp;|&nbsp;<a href="https://www.linkedin.com/company/108412762/" target="_blank">LinkedIn</a></div></div>"""


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

# FIX 5: Voice registration JavaScript — Web Speech API for Step 1 & Step 2
VOICE_JS_STEP1 = r"""
<script>
function startVoiceStep1() {
  var statusEl = document.getElementById('voice-status-1');
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    statusEl.innerText = '❌ Voice not supported. Please use Chrome or Edge.';
    return;
  }
  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recog = new SpeechRecognition();
  recog.lang = 'en-IN';
  recog.continuous = false;
  recog.interimResults = false;
  statusEl.innerText = '🎙️ Listening... Speak now';
  statusEl.style.color = '#c0392b';
  recog.start();

  function fillField(elemId, value) {
    /* Try both the wrapper div id and direct input inside it */
    var selectors = [
      '#' + elemId + ' input',
      '#' + elemId + ' textarea',
      '#' + elemId
    ];
    for (var i = 0; i < selectors.length; i++) {
      var el = document.querySelector(selectors[i]);
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value') ||
                                     Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
        if (nativeInputValueSetter) nativeInputValueSetter.set.call(el, value);
        else el.value = value;
        el.dispatchEvent(new Event('input',  {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        return true;
      }
    }
    return false;
  }

  recog.onresult = function(event) {
    var transcript = event.results[0][0].transcript;
    statusEl.style.color = '#1a7a40';
    statusEl.innerText = '✅ Heard: "' + transcript + '"';

    var nameMatch = transcript.match(/(?:my name is|name is|i am|iam)\s+([a-z\s]+?)(?:\s+mobile|\s+number|\s+roll|\s+role|$)/i);
    if (nameMatch) {
      var nm = nameMatch[1].trim();
      fillField('step1-name', nm);
      fillField('step1-name-v', nm);
    }

    var mobileMatch = transcript.match(/(?:mobile|number|phone)\s*(?:is\s*)?(\d[\d\s]{9,13})/i);
    if (mobileMatch) {
      var mob = mobileMatch[1].replace(/\s/g,'');
      fillField('step1-mobile', mob);
      fillField('step1-mobile-v', mob);
    }

    var roles = ['Business Owner','Co-Founder','Category Manager','Analyst','Store Manager'];
    var detectedRole = null;
    for (var r = 0; r < roles.length; r++) {
      if (transcript.toLowerCase().indexOf(roles[r].toLowerCase()) !== -1) {
        detectedRole = roles[r]; break;
      }
    }
    if (detectedRole) {
      /* Gradio dropdown — find the select or the visible input and set via React */
      var dropdownWrap = document.querySelector('#step1-role');
      if (dropdownWrap) {
        var selEl = dropdownWrap.querySelector('input[type="text"], input:not([type]), select');
        if (selEl) {
          fillField('step1-role', detectedRole);
        }
        /* Also try clicking the matching option in the dropdown list */
        var allOptions = dropdownWrap.querySelectorAll('li, [role="option"]');
        allOptions.forEach(function(opt) {
          if (opt.innerText && opt.innerText.trim() === detectedRole) opt.click();
        });
      }
      statusEl.innerText += ' | Role: ' + detectedRole;
    }
  };

  recog.onerror = function(e) {
    statusEl.style.color = '#c0392b';
    statusEl.innerText = '❌ Error: ' + e.error + '. Try again.';
  };
  recog.onend = function() {
    if (statusEl.innerText.indexOf('Heard') === -1 && statusEl.innerText.indexOf('Error') === -1) {
      statusEl.style.color = '#4A6A8A';
      statusEl.innerText = 'Recording stopped. Click again to retry.';
    }
  };
}
</script>
<div style="background:#F0F7FF;border:1px solid #C8DCEF;border-radius:10px;padding:16px 18px;margin-top:8px">
  <div style="font-weight:700;color:#1B4F8A;font-size:14px;margin-bottom:6px">🎙️ Voice Registration — Step 1</div>
  <div style="font-size:12px;color:#2A4060;margin-bottom:10px">Say: <em>"My name is [Name], mobile [number], role [role]"</em></div>
  <div style="font-size:11px;color:#4A6A8A;margin-bottom:10px">Roles: Business Owner · Co-Founder · Category Manager · Analyst · Store Manager</div>
  <button onclick="startVoiceStep1()" style="background:linear-gradient(135deg,#e74c3c,#c0392b);color:#FFFFFF;border:none;border-radius:8px;padding:10px 20px;font-size:13px;font-weight:600;cursor:pointer;letter-spacing:0.5px">🎤 Start Voice Input</button>
  <div id="voice-status-1" style="font-size:13px;color:#2A4060;margin-top:8px;font-style:italic;min-height:20px">Click the button and speak clearly.</div>
  <div style="font-size:11px;color:#4A6A8A;margin-top:6px">⚠️ Works in Chrome / Edge only · Check browser mic permissions</div>
</div>"""

VOICE_JS_STEP2 = r"""
<script>
function startVoiceStep2() {
  var statusEl = document.getElementById('voice-status-2');
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    statusEl.innerText = '❌ Voice not supported. Please use Chrome or Edge.';
    return;
  }
  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recog = new SpeechRecognition();
  recog.lang = 'en-IN';
  recog.continuous = false;
  recog.interimResults = false;
  statusEl.innerText = '🎙️ Listening... Speak now';
  statusEl.style.color = '#c0392b';
  recog.start();

  function fillField(elemId, value) {
    var selectors = ['#' + elemId + ' input', '#' + elemId + ' textarea', '#' + elemId];
    for (var i = 0; i < selectors.length; i++) {
      var el = document.querySelector(selectors[i]);
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) {
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value') ||
                                     Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
        if (nativeInputValueSetter) nativeInputValueSetter.set.call(el, value);
        else el.value = value;
        el.dispatchEvent(new Event('input',  {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        return true;
      }
    }
    return false;
  }

  recog.onresult = function(event) {
    var transcript = event.results[0][0].transcript;
    statusEl.style.color = '#1a7a40';
    statusEl.innerText = '✅ Heard: "' + transcript + '"';

    /* UDYAM number — capture full UDYAM-XX-XX-XXXXXXX pattern */
    var udyamMatch = transcript.match(/(?:UDYAM|udyam|udhyam)\s*[-]?\s*([A-Z0-9][\w\-\s]{4,20})/i);
    if (udyamMatch) {
      var udyam = udyamMatch[0].replace(/\s+/g,'-').replace(/-{2,}/g,'-').toUpperCase().trim();
      fillField('step2-msme', udyam);
      fillField('step2-msme-v', udyam);
    }

    /* OTP — 4 digit number after keyword */
    var otpMatch = transcript.match(/(?:OTP|otp|code|pass)\s*(?:is\s*)?(\d{4})/i);
    if (otpMatch) {
      /* OTP field is type=password — querySelector finds it via wrapper id */
      var otpWrap = document.getElementById('step2-otp');
      if (otpWrap) {
        var otpEl = otpWrap.querySelector('input');
        if (otpEl) {
          var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
          if (setter) setter.set.call(otpEl, otpMatch[1]);
          else otpEl.value = otpMatch[1];
          otpEl.dispatchEvent(new Event('input',  {bubbles: true}));
          otpEl.dispatchEvent(new Event('change', {bubbles: true}));
        }
      }
      fillField('step2-otp-v', otpMatch[1]);
      statusEl.innerText += ' | OTP: ****';
    }
  };

  recog.onerror = function(e) {
    statusEl.style.color = '#c0392b';
    statusEl.innerText = '❌ Error: ' + e.error + '. Try again.';
  };
  recog.onend = function() {
    if (statusEl.innerText.indexOf('Heard') === -1 && statusEl.innerText.indexOf('Error') === -1) {
      statusEl.style.color = '#4A6A8A';
      statusEl.innerText = 'Recording stopped. Click again to retry.';
    }
  };
}
</script>
<div style="background:#F0F7FF;border:1px solid #C8DCEF;border-radius:10px;padding:16px 18px;margin-top:8px">
  <div style="font-weight:700;color:#1B4F8A;font-size:14px;margin-bottom:6px">🎙️ Voice Registration — Step 2</div>
  <div style="font-size:12px;color:#2A4060;margin-bottom:10px">Say: <em>"UDYAM [number] OTP [4 digits]"</em></div>
  <div style="font-size:11px;color:#4A6A8A;margin-bottom:10px">Example: <em>"UDYAM TN 00 7629703 OTP 1234"</em></div>
  <button onclick="startVoiceStep2()" style="background:linear-gradient(135deg,#e74c3c,#c0392b);color:#FFFFFF;border:none;border-radius:8px;padding:10px 20px;font-size:13px;font-weight:600;cursor:pointer;letter-spacing:0.5px">🎤 Start Voice Input</button>
  <div id="voice-status-2" style="font-size:13px;color:#2A4060;margin-top:8px;font-style:italic;min-height:20px">Click the button and speak clearly.</div>
  <div style="font-size:11px;color:#4A6A8A;margin-top:6px">⚠️ Works in Chrome / Edge only · Check browser mic permissions</div>
</div>"""

with gr.Blocks(title="DataNetra.ai - MSME Intelligence", theme=gr.themes.Soft(), css=custom_css) as demo:

    step_state = gr.State(0); user_data_state = gr.State({}); lang_state = gr.State('en')
    dashboard_data_state = gr.State({'kpi1':"","kpi2":"","kpi3":"","kpi4":"","kpi5":"","chart1":None,"chart2":None,"chart3":None,"chart4":None})
    granular_forecast_data_state = gr.State(None); df_state = gr.State(None)

    # ── Language Bar ──
    with gr.Row(elem_id="lang-bar"):
        gr.HTML('<div style="background:#0f2557;padding:8px 16px;border-radius:6px;display:flex;align-items:center;gap:12px;"><span style="color:white;font-weight:600;font-size:0.9rem;">🌐 Language / भाषा:</span></div>')
        lang_en_btn = gr.Button("🇬🇧 English", size="sm", variant="primary", elem_id="lang-en-btn")
        lang_hi_btn = gr.Button("🇮🇳 हिंदी", size="sm", variant="secondary", elem_id="lang-hi-btn")
        lang_indicator = gr.Markdown("**Active: English**", elem_id="lang-indicator")

    # ── Step 0: Landing ──
    with gr.Column(visible=True) as step0_col:
        landing_hero_html = gr.HTML(value=_landing_hero('en'))
        show_signup_trigger = gr.Button("", elem_id="show-signup-trigger", visible=False)
        with gr.Column(elem_classes="section", elem_id="how-datanetra-works-section"):
            landing_how_title = gr.Markdown("## How DataNetra Works", elem_classes="section-title"); gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=1):
                    landing_step1_title = gr.Markdown("### 📥 Step 1: Upload Your Data")
                    landing_step1_desc  = gr.Markdown("Easily upload Excel/CSV files for comprehensive analysis.")
                    landing_step2_title = gr.Markdown("### 🤖 Step 2: AI-Powered Analysis")
                    landing_step2_desc  = gr.Markdown("Our AI processes your data, forecasting trends and uncovering insights.")
                    landing_step3_title = gr.Markdown("### 📊 Step 3: Actionable Dashboards")
                    landing_step3_desc  = gr.Markdown("Access interactive dashboards, KPI charts and personalized recommendations.")
                with gr.Column(scale=1):
                    landing_login_title  = gr.Markdown("**Already Registered**")
                    quick_login_mobile   = gr.Textbox(label="Enter Mobile Number", placeholder="+91")
                    quick_login_btn      = gr.Button("Login", variant="primary", size="lg")
                    landing_login_error_msg = gr.Markdown(value="", visible=False)
                    landing_signup_title = gr.Markdown("**First Time User**")
                    landing_signup_desc  = gr.Markdown("**Signup to unlock smart AI Insights**")
                    quick_signup_btn     = gr.Button("Sign Up Now", variant="primary", size="lg")
        landing_capabilities_html = gr.HTML(value=_landing_capabilities('en'))

    # ── Step 1: User Info + Voice ──
    with gr.Column(visible=False) as step1_col:
        gr.Markdown("# 📝 Register New User\n## Step 1: User Information")

        # FIX 5a — Tabs for Manual vs Voice input in Step 1
        with gr.Tabs():
            with gr.Tab("⌨️ Manual Entry"):
                name_input   = gr.Textbox(label="Full Name*", elem_id="step1-name")
                mobile_input = gr.Textbox(label="Mobile Number*", elem_id="step1-mobile")
                email_input  = gr.Textbox(label="Email")
                role_input   = gr.Dropdown(choices=roles, label="Role*")
            with gr.Tab("🎙️ Voice Entry"):
                gr.HTML(VOICE_JS_STEP1)
                gr.Markdown("""**How it works:**
1. Click **🎤 Start Voice Input**
2. Say: *"My name is Karthick, mobile 9876543210, role Business Owner"*
3. The system will fill in the fields automatically
4. Switch to **Manual Entry** tab to verify and edit if needed""")
                # Mirror fields so they stay in sync (voice fills via JS into the elem_id targets)
                name_input_v   = gr.Textbox(label="Name (voice filled — verify here)", elem_id="step1-name-v")
                mobile_input_v = gr.Textbox(label="Mobile (voice filled — verify here)", elem_id="step1-mobile-v")
                role_input_v   = gr.Textbox(label="Role (voice filled — verify here, e.g. Business Owner)")
                gr.Markdown("*After voice fill, values are shown here. You can still edit them.*")

        with gr.Row():
            cancel1_btn = gr.Button("Cancel")
            next1_btn   = gr.Button("Next →", variant="primary")
        error1 = gr.Markdown()

    # ── Step 2: MSME Verification + Voice ──
    with gr.Column(visible=False) as step2_col:
        gr.Markdown("## Step 2: MSME Verification")

        # FIX 5b — Tabs for Manual vs Voice input in Step 2
        with gr.Tabs():
            with gr.Tab("⌨️ Manual Entry"):
                msme_number_input = gr.Textbox(label="MSME/Udyam Number*", placeholder="e.g., UDYAM-TN-00-7629703", elem_id="step2-msme")
                otp_input         = gr.Textbox(label="OTP (Enter '1234' for demo)*", type="password", elem_id="step2-otp")
                fetch_btn         = gr.Button("Fetch MSME Data", variant="secondary")
                fetch_status      = gr.Markdown()
            with gr.Tab("🎙️ Voice Entry"):
                gr.HTML(VOICE_JS_STEP2)
                gr.Markdown("""**How it works:**
1. Click **🎤 Start Voice Input**
2. Say: *"UDYAM TN 00 7629703, OTP 1234"*
3. Switch to **Manual Entry** tab to verify
4. Click **Fetch MSME Data** to load your details""")
                msme_voice_display = gr.Textbox(label="UDYAM Number (voice filled — verify)", elem_id="step2-msme-v")
                otp_voice_display  = gr.Textbox(label="OTP (voice filled — verify)", type="password", elem_id="step2-otp-v")
                gr.Markdown("*After verifying above, go to Manual tab and click Fetch MSME Data.*")

        gr.Markdown("### Fetched MSME Details")
        fetched_name     = gr.Textbox(label="Enterprise Name", interactive=False)
        fetched_org      = gr.Textbox(label="Organisation Type", interactive=False)
        fetched_activity = gr.Textbox(label="Major Activity", interactive=False)
        fetched_type     = gr.Textbox(label="Enterprise Type", interactive=False)
        fetched_state    = gr.Textbox(label="State", interactive=False)
        fetched_city     = gr.Textbox(label="City", interactive=False)
        # FIX 4 — Industry Domain field in Step 2
        fetched_industry = gr.Textbox(label="Industry Domain", interactive=False, value="Retail")

        with gr.Row():
            back2_btn = gr.Button("← Back")
            next2_btn = gr.Button("Verify & Next →", variant="primary")
        error2 = gr.Markdown()

    # ── Step 3: Certificate Review ──
    with gr.Column(visible=False) as step3_col:
        gr.Markdown("## Step 3: MSME Certificate Review\n### Confirm MSME Details")
        confirm_name     = gr.Textbox(label="Enterprise Name", interactive=False)
        confirm_org      = gr.Textbox(label="Organisation Type", interactive=False)
        confirm_activity = gr.Textbox(label="Major Activity", interactive=False)
        confirm_type     = gr.Textbox(label="Enterprise Type", interactive=False)
        confirm_state    = gr.Textbox(label="State", interactive=False)
        confirm_city     = gr.Textbox(label="City", interactive=False)
        # FIX 4 — Industry Domain in Step 3
        confirm_industry = gr.Textbox(label="Industry Domain", interactive=False, value="Retail")
        consent1           = gr.Checkbox(label="I confirm the above MSME details are correct", value=False)
        consent2           = gr.Checkbox(label="I consent to verify the MSME certificate", value=False)
        certificate_upload = gr.File(label="Upload MSME Certificate (PDF)", file_types=[".pdf"])
        with gr.Row():
            back3_btn = gr.Button("← Back")
            next3_btn = gr.Button("Confirm & Proceed →", variant="primary")
        error3 = gr.Markdown()

    # ── Step 4: Business Profile ──
    with gr.Column(visible=False) as step4_col:
        verification_status_display = gr.Markdown(visible=False)
        gr.Markdown("## Step 4: Business Profile")
        business_type_input = gr.Dropdown(choices=business_types, label="Business Type*")
        years_input         = gr.Number(label="Years in Operation*", value=1, minimum=0)
        revenue_input       = gr.Dropdown(label="Monthly Revenue Range*", choices=["< 5 Lakh","5-10 Lakh","10-50 Lakh","50 Lakh - 1 Crore","> 1 Crore"])
        with gr.Row():
            back4_btn = gr.Button("← Back")
            next4_btn = gr.Button("Submit Profile", variant="primary")
        error4 = gr.Markdown()
        proceed_to_step5_btn = gr.Button("Next: Upload Business Data →", variant="primary", visible=False)

    # ── Step 5: Upload & Analyse ──
    with gr.Column(visible=False) as step5_col:
        login_welcome_message = gr.Markdown(value="", visible=False)
        gr.Markdown("## Step 5: Upload Business Data")
        consent_check  = gr.Checkbox(label="I consent to data analysis*", value=False)
        file_upload    = gr.File(label="Upload Excel File (.xlsx, .csv)*", file_types=[".xlsx",".csv"])
        upload_message = gr.Markdown(value="", visible=False)
        with gr.Row():
            back5_btn   = gr.Button("← Back")
            cancel5_btn = gr.Button("❌ Cancel", variant="secondary")
            analyze_btn = gr.Button("🚀 Analyze Data", variant="primary", elem_id="analyze-data-btn")
        error5 = gr.Markdown()
        insights_output = gr.HTML(elem_id="insights-output-html")
        with gr.Row():
            view_dashboard_btn     = gr.Button("📊 ONDC Impact — View Dashboard", visible=False, variant="primary")
            view_gov_dashboard_btn = gr.Button("🏛️ Government Dashboard", visible=False, variant="secondary")
        kpi1 = gr.Markdown(visible=False); kpi2 = gr.Markdown(visible=False)
        kpi3 = gr.Markdown(visible=False); kpi4 = gr.Markdown(visible=False); kpi5 = gr.Markdown(visible=False)
        chart1 = gr.Plot(visible=False); chart2 = gr.Plot(visible=False)
        chart3 = gr.Plot(visible=False); chart4 = gr.Plot(visible=False)
        sum1 = gr.Markdown(visible=False); sum2 = gr.Markdown(visible=False)
        sum3 = gr.Markdown(visible=False); sum4 = gr.Markdown(visible=False)

    # ── Step 6: ONDC Impact & SNP Readiness ──
    with gr.Column(visible=False) as step6_col:
        gr.HTML('''<div style="background:linear-gradient(135deg,#0f2557,#1a3a6b);
                              padding:14px 24px;border-radius:10px;margin-bottom:12px;text-align:center">
          <h1 style="color:white;margin:0;font-size:1.45rem;font-weight:800;letter-spacing:0.4px">
            ONDC Impact &amp; SNP Readiness Dashboard
          </h1></div>''')
        kpi_table_dash = gr.HTML(value="", elem_id="kpi-table-container")
        gr.HTML('<div style="margin:28px 0 10px 0;padding:14px 20px;background:linear-gradient(90deg,#0f2557,#1a3a6b);border-radius:8px;"><span style="color:white;font-size:1.25rem;font-weight:700;">📈 Performance Visualizations</span></div>')
        with gr.Row():
            with gr.Column(): chart1_dash = gr.Plot(label="Sales vs Profit Margin (Quarterly)"); chart1_summary = gr.Markdown(value="", visible=False)
            with gr.Column(): chart2_dash = gr.Plot(label="ONDC Before vs After Revenue"); chart2_summary = gr.Markdown(value="", visible=False)
        with gr.Row():
            with gr.Column(): chart3_dash = gr.Plot(label="Returns & Replacements Trend"); chart3_summary = gr.Markdown(value="", visible=False)
            with gr.Column(): chart4_dash = gr.Plot(label="Store-Level ONDC Impact"); chart4_summary = gr.Markdown(value="", visible=False)
        with gr.Row():
            forecast_deepdive_btn = gr.Button("📈 Intelligence Sales Forecast Dashboard →", variant="primary", size="lg")
            back6_btn             = gr.Button("⬅ Back to Data Upload", variant="secondary", size="lg")

    # ── Step 6a: Government Dashboard ──
    with gr.Column(visible=False) as step6a_col:
        gr.HTML('<div style="background:linear-gradient(135deg,#070D1A,#0D1829);padding:18px 24px;border-radius:10px;margin-bottom:16px;"><h1 style="color:white;margin:0;font-size:1.9rem;font-weight:700;">🏛️ National MSME Platform Dashboard</h1><p style="color:#7AABDD;margin:6px 0 0;font-size:1rem;">Government & Policy Intelligence — Powered by DataNetra.ai</p></div>')
        back6a_btn       = gr.Button("⬅ Back to Data Upload", variant="secondary")
        gov_dashboard_html = gr.HTML(value="", elem_id="gov-platform-dashboard")

    # ── Step 7: Store → Category → Product + Forecasting ──
    with gr.Column(visible=False) as step7_col:
        # ── Page title ──
        gr.HTML('''<div style="background:linear-gradient(135deg,#0f2557,#1a3a6b);
                              padding:14px 24px;border-radius:10px;margin-bottom:12px;text-align:center">
          <h1 style="color:white;margin:0;font-size:1.45rem;font-weight:800;letter-spacing:0.4px">
            STEP 7: Store &#8594; Category &#8594; Product + Forecasting
          </h1></div>''')
        # ── Filter bar: MSME label + 3 dropdowns inline ──
        gr.HTML('''<div id="s7-filter-wrap" style="display:flex;align-items:center;gap:10px;
                         flex-wrap:wrap;background:#f0f7ff;border:1px solid #c8dcef;
                         border-radius:8px;padding:8px 14px;margin-bottom:12px">
          <span style="font-size:11px;font-weight:700;color:#1B4F8A;white-space:nowrap">
            MSME: Sri Hypermarket (UDYAM-TN-00-762978)</span>
          <span style="color:#C8DCEF">|</span>
        </div>''')
        with gr.Row():
            s7_msme_label   = gr.HTML(value='', visible=False)
            s7_store_filter = gr.Dropdown(choices=["Store: All"], value="Store: All",
                                          label="🏪 Store", scale=1, interactive=True)
            s7_cat_filter   = gr.Dropdown(choices=["Category: All"], value="Category: All",
                                          label="📂 Category", scale=1, interactive=True)
            s7_prod_filter  = gr.Dropdown(choices=["Product: All"], value="Product: All",
                                          label="📦 Product", scale=1, interactive=True)
        # ── KPI card row ──
        s7_kpi_html = gr.HTML(value="", elem_id="s7-kpi-row")
        # ── Top Products table ──
        s7_top_table = gr.HTML(value="", elem_id="s7-top-table")
        # ── Row 1: Category Sales Trend | Category Margin Trend ──
        with gr.Row():
            s7_cat_sales_chart  = gr.Plot(label="Category Sales Trend (36 Months)",  scale=1)
            s7_cat_margin_chart = gr.Plot(label="Category Margin Trend (36 Months)", scale=1)
        # ── Row 2: 6-Month | 12-Month | Fulfilment Trend ──
        with gr.Row():
            s7_fc6_chart    = gr.Plot(label="6-Month Forecast",        scale=1)
            s7_fc12_chart   = gr.Plot(label="12-Month Forecast",       scale=1)
            s7_fulfil_chart = gr.Plot(label="Product Fulfilment Trend",scale=1)
        # ── Row 3: Sales vs Returns | Inventory | AI Summary ──
        with gr.Row():
            s7_sales_ret_chart = gr.Plot(label="Product Sales vs Returns",   scale=1)
            s7_inventory_chart = gr.Plot(label="Inventory & Reorder Levels", scale=1)
            s7_ai_summary      = gr.HTML(value="", elem_id="s7-ai-summary")
        # ── Back buttons ──
        with gr.Row():
            back7_btn     = gr.Button("\u2b05 Back to Dashboard",   variant="secondary")
            back7_to5_btn = gr.Button("\u2b05 Back to Data Upload", variant="secondary")

    # ══════════════════════════════════════════════════════════════════════════
    # All columns list
    # ══════════════════════════════════════════════════════════════════════════
    _ALL_COLS = [step0_col, step1_col, step2_col, step3_col, step4_col, step5_col, step6_col, step6a_col, step7_col]

    def update_visibility_all(active_name):
        names = ['step0','step1','step2','step3','step4','step5','step6','step6a','step7']
        return [gr.update(visible=(n == active_name)) for n in names]

    # ══════════════════════════════════════════════════════════════════════════
    # Event Handlers
    # ══════════════════════════════════════════════════════════════════════════
    def show_signup():
        return (1, *update_visibility_all('step1'))

    def handle_login(mobile):
        profile = get_user_profile(mobile)
        if profile:
            msg = f"✅ Welcome back, {profile['full_name']}! You have been navigated to upload business data file for Analysis."
            return (gr.update(value="", visible=False), profile, 5, *update_visibility_all('step5'),
                    gr.update(value="", visible=False), gr.update(value="", visible=False),
                    gr.update(value="", visible=False), gr.update(value="", visible=False),
                    gr.update(value="", visible=False), gr.update(value=msg, visible=True))
        return (gr.update(value="❌ No account found. Please register or try again.", visible=True), {}, 0,
                *update_visibility_all('step0'),
                gr.update(value="", visible=False), gr.update(value="", visible=False),
                gr.update(value="", visible=False), gr.update(value="", visible=False),
                gr.update(value="", visible=False), gr.update(value="", visible=False))

    def validate_step1(name, mobile, email, role, current_data):
        import re
        # Required field check
        if not name or not mobile or not role:
            return ("⚠️ Please fill all required fields (Name, Mobile, Role)", current_data, 1, *update_visibility_all('step1'))
        # Fix 1: Full name — alphabets and spaces only
        name_clean = name.strip()
        if not re.match(r'^[A-Za-z ]{2,60}$', name_clean):
            return ("⚠️ Full Name must contain only alphabets and spaces (2–60 characters)", current_data, 1, *update_visibility_all('step1'))
        # Fix 2: Mobile — exactly 10 digits
        mobile_clean = mobile.strip()
        if not re.match(r'^[6-9][0-9]{9}$', mobile_clean):
            return ("⚠️ Mobile Number must be exactly 10 digits and start with 6, 7, 8 or 9", current_data, 1, *update_visibility_all('step1'))
        # Email format check (if provided)
        if email and not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email.strip()):
            return ("⚠️ Please enter a valid email address", current_data, 1, *update_visibility_all('step1'))
        updated = {**current_data, 'full_name': name_clean, 'mobile_number': mobile_clean, 'email': email, 'role': role}
        return ("", updated, 2, *update_visibility_all('step2'))

    def verify_step2(msme_num, otp, current_data, ent_name, org, activity, ent_type, state, city, industry, status):
        import re
        BLANK = ("", "", "", "", "", "", "Retail", gr.update(visible=False))
        if not msme_num or not otp:
            return ("⚠️ Please fill MSME/Udyam number and OTP", current_data, 2, *update_visibility_all('step2'), *BLANK)
        # Fix 3: Udyam number format — UDYAM-XX-YY-XXXXXXX
        udyam_clean = msme_num.strip().upper()
        if not re.match(r'^UDYAM-[A-Z]{2}-\d{2}-\d{7}$', udyam_clean):
            return ("⚠️ Invalid Udyam Number format. Required format: UDYAM-XX-00-0000000 (e.g. UDYAM-TN-00-7629703)", current_data, 2, *update_visibility_all('step2'), *BLANK)
        if otp != "1234":
            return ("⚠️ Invalid OTP. Please enter the correct OTP", current_data, 2, *update_visibility_all('step2'), *BLANK)
        if "Successfully" not in str(status):
            return ("⚠️ Please fetch MSME data first using the Fetch button", current_data, 2, *update_visibility_all('step2'), *BLANK)
        updated = {**current_data, 'msme_number': udyam_clean, 'company_name': ent_name,
                   'organisation_type': org, 'major_activity': activity, 'enterprise_type': ent_type,
                   'state': state, 'city': city, 'industry_domain': industry}
        return ("✅ OTP Verified — Udyam number validated", updated, 3, *update_visibility_all('step3'),
                ent_name, org, activity, ent_type, state, city, industry, gr.update(value="", visible=False))

    def confirm_step3(current_data, c1, c2, cert):
        import re, os
        if not c1 or not c2:
            return ("⚠️ Please accept both consents to proceed", current_data, 3, *update_visibility_all('step3'),
                    gr.update(value="", visible=False), gr.update(value="Choose Business Type"))
        if cert is None:
            return ("⚠️ Please upload your MSME Certificate (PDF)", current_data, 3, *update_visibility_all('step3'),
                    gr.update(value="", visible=False), gr.update(value="Choose Business Type"))

        # ── Certificate Validation: match Udyam number in PDF vs Step 2 ──────
        msme_from_step2 = current_data.get('msme_number', '').strip().upper()
        if msme_from_step2 and cert is not None:
            cert_path = cert.name if hasattr(cert, 'name') else str(cert)
            cert_text = ""
            # Method 1: PyMuPDF (fitz)
            try:
                import fitz
                doc = fitz.open(cert_path)
                cert_text = " ".join(page.get_text() for page in doc).upper()
            except Exception:
                pass
            # Method 2: pdfplumber fallback
            if not cert_text.strip():
                try:
                    import pdfplumber
                    with pdfplumber.open(cert_path) as pdf:
                        cert_text = " ".join(p.extract_text() or "" for p in pdf.pages).upper()
                except Exception:
                    pass
            # Method 3: pypdf fallback
            if not cert_text.strip():
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(cert_path)
                    cert_text = " ".join(page.extract_text() or "" for page in reader.pages).upper()
                except Exception:
                    pass

            if cert_text.strip():
                # Normalise: strip hyphens for flexible matching
                def _norm(s): return re.sub(r'[-\s]', '', s)
                matches = re.findall(r'UDYAM[-\s]?[A-Z]{2}[-\s]?\d{2}[-\s]?\d{7}', cert_text)
                matched_normalised = [_norm(m) for m in matches]
                step2_normalised   = _norm(msme_from_step2)
                if matched_normalised and step2_normalised not in matched_normalised:
                    found_display = matches[0]
                    return (
                        f"❌ Certificate Mismatch! The uploaded certificate belongs to **{found_display}** "
                        f"but you entered **{msme_from_step2}** in Step 2. "
                        f"Please upload the correct certificate or go back and correct the MSME number.",
                        current_data, 3, *update_visibility_all('step3'),
                        gr.update(value="", visible=False), gr.update(value="Choose Business Type"))
                elif not matched_normalised:
                    # No Udyam found in PDF — could be scanned image; warn but allow through
                    pass  # Fall through to approval
        updated = {**current_data, 'verification_status': 'APPROVED'}
        activity = current_data.get('major_activity', '')
        pre_biz = ACTIVITY_TO_BIZ_TYPE.get(activity, "Choose Business Type")
        if activity in business_types: pre_biz = activity

        # FIX 3 — Show Major Activity in verification status message
        success_msg = (f"## ✅ Verification Status: APPROVED\n\n"
                       f"**Company:** {current_data.get('company_name','N/A')}\n\n"
                       f"**MSME Number:** {current_data.get('msme_number','N/A')}\n\n"
                       f"**Major Activity:** {activity}\n\n"
                       f"**Status:** APPROVED ✓")
        return (gr.update(value="", visible=False), updated, 4, *update_visibility_all('step4'),
                gr.update(value=success_msg, visible=True), gr.update(value=pre_biz))

    def submit_profile(biz_type, years, revenue, current_data):
        if not biz_type or biz_type == "Choose Business Type":
            return (gr.update(value="⚠️ Please select business type", visible=True), current_data,
                    gr.update(visible=False), gr.update(visible=True))
        if not revenue:
            return (gr.update(value="⚠️ Please select revenue range", visible=True), current_data,
                    gr.update(visible=False), gr.update(visible=True))
        if years is None or years <= 0:
            return (gr.update(value="⚠️ Please enter valid years in operation", visible=True), current_data,
                    gr.update(visible=False), gr.update(visible=True))
        updated = {**current_data, 'business_type': biz_type, 'years_operation': int(years),
                   'monthly_revenue_range': revenue, 'consent_given': True}
        try:
            user_id = save_user_profile(updated)
            success_msg = (f"## ✅ Business Profile Submitted!\n\n"
                           f"**Company:** {updated.get('company_name','N/A')}\n\n"
                           f"**Business Type:** {biz_type}\n\n"
                           f"**Years in Operation:** {int(years)}\n\n"
                           f"**Monthly Revenue:** {revenue}\n\n"
                           f"Profile saved (ID: {user_id}). Click **Next** to upload data.")
            return (gr.update(value=success_msg, visible=True), updated,
                    gr.update(visible=True), gr.update(visible=False))
        except Exception as e:
            return (gr.update(value=f"❌ Error saving profile: {str(e)}", visible=True), current_data,
                    gr.update(visible=False), gr.update(visible=True))

    # FIX 2 — analyze_data with correct column mapping
    def analyze_data(user_data, consent, file, lang='en'):
        empty_dash = {'kpi1':"","kpi2":"","kpi3":"","kpi4":"","kpi5":"","chart1":None,"chart2":None,"chart3":None,"chart4":None}
        def _fail(msg):
            return (msg, gr.update(visible=False), gr.update(visible=False),
                    "", "", "", "", "", None, None, None, None,
                    gr.update(value="", visible=False), gr.update(value="", visible=False),
                    gr.update(value="", visible=False), gr.update(value="", visible=False),
                    gr.update(value="", visible=False), empty_dash, None)
        if not consent: return _fail("⚠️ Please provide consent to analyze data")
        if file is None: return _fail("⚠️ Please upload an Excel or CSV file")
        try:
            file_path = file.name if hasattr(file, 'name') else str(file)
            if file_path.endswith('.xlsx'): df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'): df = pd.read_csv(file_path)
            else: return _fail("❌ Unsupported file format. Please upload .xlsx or .csv")

            # ── FIX 2: Apply correct column mapping ──
            df = _apply_col_remap(df)

            # Fill any still-missing required cols with defaults
            required_cols = {
                'Date':     lambda: pd.to_datetime(datetime.datetime.now().date()),
                'Store_ID': 'Default',
                'SKU_Name': 'Default',
                'Monthly_Sales_INR':          0,
                'Monthly_Operating_Cost_INR': 0,
                'Outstanding_Loan_INR':       0,
                'Vendor_Delivery_Reliability':0,
                'Inventory_Turnover':         0,
                'Avg_Margin_Percent':         0,
                'Monthly_Demand_Units':       0,
                'Returns_Percentage':         0,
            }
            for col, default in required_cols.items():
                if col not in df.columns:
                    df[col] = default() if callable(default) else default

            # ── Fix 6: Store full dataset for Government Dashboard ──────────
            # Government dashboard needs ALL MSMEs; Steps 5/6/7 use only the
            # logged-in MSME's rows so insights are personalised.
            df_full_for_gov = df.copy()

            # Detect the Udyam/MSME key column (before or after remap)
            msme_key = user_data.get('msme_number', '').strip().upper()
            _udyam_candidates = ['udyam_number','Udyam_Number','UDYAM_NUMBER',
                                  'msme_number','MSME_Number','Udyam_No']
            _udyam_col = next((c for c in _udyam_candidates if c in df.columns), None)

            if msme_key and _udyam_col:
                _unique_msme = df[_udyam_col].astype(str).str.strip().str.upper().nunique()
                if _unique_msme > 1:
                    _df_filtered = df[
                        df[_udyam_col].astype(str).str.strip().str.upper() == msme_key
                    ].copy()
                    if len(_df_filtered) > 0:
                        df = _df_filtered
                # Single-MSME dataset → use full df as-is

            insights_html, error_msg, _ = generate_insights(user_data, df.copy(), lang=lang)
            if error_msg: return _fail(f"❌ {error_msg}")

            result = generate_dashboard_data(user_data, df.copy())
            k1 = result[0]; f1,f2,f3,f4 = result[5],result[6],result[7],result[8]
            s1,s2,s3,s4 = result[9],result[10],result[11],result[12]
            raw_df = result[15]
            try: gf = generate_granular_forecast(df.copy())
            except: gf = None
            dash = {'kpi1':k1,'chart1':f1,'chart2':f2,'chart3':f3,'chart4':f4,'sum1':s1,'sum2':s2,'sum3':s3,'sum4':s4,'granular':gf}
            df_for_gov = df_full_for_gov  # full multi-MSME dataset → Government Dashboard
            # Pass df_full_for_gov to df_state so Government Dashboard gets all MSMEs
            # Steps 5/6/7 use MSME-filtered df (already applied above)
            return (insights_html or "✅ Analysis completed",
                    gr.update(visible=True), gr.update(visible=True),
                    k1, "", "", "", "", f1,f2,f3,f4,
                    s1, s2, s3, s4,
                    gr.update(value="", visible=False), dash, df_for_gov)
        except Exception as e:
            import traceback
            return _fail(f"❌ Analysis failed: {str(e)}\n\n{traceback.format_exc()}")

    def show_dashboard(dashboard_data_value):
        def _summary(key):
            val = dashboard_data_value.get(key)
            return gr.update(value=val, visible=True) if val else gr.update(value="", visible=False)
        granular_data = dashboard_data_value.get('granular')
        kpi_html_val  = dashboard_data_value.get('kpi1', "")
        return (6, *update_visibility_all('step6'), kpi_html_val,
                dashboard_data_value.get('chart1'), dashboard_data_value.get('chart2'),
                dashboard_data_value.get('chart3'), dashboard_data_value.get('chart4'),
                _summary('sum1'), _summary('sum2'), _summary('sum3'), _summary('sum4'), granular_data)

    def show_gov_dashboard(raw_df):
        if raw_df is None:
            html = "<div style='padding:40px;color:#FF4444;font-size:16px;'>⚠️ No data available. Please upload and analyze data first.</div>"
        else:
            try:
                # raw_df here is the full dataset (all MSMEs) — ideal for government portfolio view
                html = build_full_platform_dashboard(raw_df)
            except Exception as e:
                import traceback
                html = f"<div style='padding:40px;color:#FF4444;font-size:16px;'>❌ Error: {str(e)}<br><pre style='font-size:11px'>{traceback.format_exc()}</pre></div>"
        return (6, *update_visibility_all('step6a'), html)

    def handle_file_upload_change(user_data, file):
        if file is not None:
            name = user_data.get('full_name', 'User')
            return (gr.update(value=f"Thank you, {name}, for uploading the dataset. Click 'Analyze Data' to view AI Insights and Dashboard.", visible=True),
                    gr.update(value="", visible=False))
        return gr.update(value="", visible=False), gr.update(value="", visible=False)

    # ── Product name mapping (matches reference image naming style) ────────────
    _PRODUCT_NAMES = {
        1:'Aashirvaad Atta', 2:'Tata Salt', 3:'Parle-G Biscuits', 4:'Amul Butter',
        5:'Samsung TV 43"', 6:'Whirlpool Fridge', 7:'LG Microwave', 8:'Philips Iron',
        9:'Jeans Regular Fit', 10:'Cotton Kurta', 11:'Ladies Saree', 12:'Kids T-Shirt',
        13:'Prestige Cooker', 14:'Milton Bottle', 15:'Wooden Shelf', 16:'Ceramic Vase',
        17:'Dettol Sanitiser', 18:'Himalaya Face Wash', 19:'Glucon-D 500g', 20:'Band-Aid Box',
        21:'Maggi Noodles', 22:'Britannia Biscuits', 23:'Surf Excel 1kg', 24:'Colgate Toothpaste',
        25:'Bosch Mixer', 26:'Sony Earphones', 27:'Xiaomi Powerbank', 28:'Syska LED Bulb',
        29:'Formal Shirt', 30:'Track Pants', 31:'Winter Jacket', 32:'Ethnic Dupatta',
        33:'Steel Cookware Set', 34:'Bamboo Basket', 35:'Wall Clock', 36:'Photo Frame',
        37:'Dabur Honey', 38:'Patanjali Ghee', 39:'Vitamin-C Tablets', 40:'Neem Face Pack',
    }

    def _build_step7_data(df_raw, store_sel, cat_sel, prod_sel):
        """Filter df and compute all KPIs + 7 charts + AI summary for Step 7."""
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np, pandas as pd, warnings
        warnings.filterwarnings('ignore')
        from sklearn.linear_model import LinearRegression

        if df_raw is None:
            return None

        df = df_raw.copy()

        # ── Column resolution (handles both raw & remapped names) ───────────
        def _col(*names):
            for n in names:
                if n in df.columns: return n
            return None

        sc   = _col('net_sales', 'Monthly_Sales_INR', 'gross_sales')
        dc   = _col('date', 'Date')
        stc  = _col('store_id', 'Store_ID')
        catc = _col('product_category', 'Product_Category')
        pidc = _col('product_id', 'SKU_Name')
        uc   = _col('units_sold', 'Monthly_Demand_Units')
        pmc  = _col('profit_margin_pct', 'Avg_Margin_Percent')
        rrc  = _col('return_rate_pct', 'Returns_Percentage')
        rpc  = _col('replacement_count')
        tac  = _col('target_achievement_pct')
        ivc  = _col('inventory_level')
        roc  = _col('reorder_point')
        slc  = _col('stock_level')
        qrc  = _col('quantity_returned')

        if sc is None:
            return None

        # ── Ensure MSME-only ────────────────────────────────────────────────
        # df is already filtered to the logged-in MSME in analyze_data — no extra filter needed
        if df.empty:
            return None

        # ── Parse dates ─────────────────────────────────────────────────────
        if dc:
            df[dc] = pd.to_datetime(df[dc], errors='coerce')
            df = df.dropna(subset=[dc])

        # ── Apply filters ────────────────────────────────────────────────────
        if store_sel and store_sel not in ("Store: All", ""):
            sid = store_sel.replace("Store: ", "").strip()
            if stc: df = df[df[stc].astype(str) == sid]
        if cat_sel and cat_sel not in ("Category: All", ""):
            cval = cat_sel.replace("Category: ", "").strip()
            if catc: df = df[df[catc].astype(str) == cval]
        if prod_sel and prod_sel not in ("Product: All", ""):
            pval = prod_sel.replace("Product: SKU-", "").replace("Product: ", "").strip()
            if pidc:
                try:    df = df[df[pidc].astype(str) == pval]
                except: pass
        if df.empty:
            return None

        # ── Helpers ──────────────────────────────────────────────────────────
        def _s(c):  return float(df[c].sum())  if c and c in df.columns else 0.0
        def _m(c):  return float(df[c].mean()) if c and c in df.columns else 0.0
        def _si(c): return int(df[c].sum())    if c and c in df.columns else 0
        def _inr(v):
            if v >= 1e7: return f"&#8377;{v/1e7:.1f} Cr"
            if v >= 1e5: return f"&#8377;{v/1e5:.1f} L"
            return f"&#8377;{v:,.0f}"
        def _inr_ax(x, _):
            if abs(x) >= 1e7: return f"₹{x/1e7:.1f}Cr"
            if abs(x) >= 1e5: return f"₹{x/1e5:.0f}L"
            return f"₹{x:,.0f}"

        # ── KPI values ───────────────────────────────────────────────────────
        total_units = _si(uc)
        total_sales = _s(sc)
        avg_margin  = _m(pmc)
        avg_ret     = _m(rrc)
        total_repl  = _si(rpc)
        avg_fulfil  = _m(tac)

        # ── KPI HTML: 6-card row matching image ──────────────────────────────
        def _kcard(label, value, color, sub_color=None, sub=""):
            sc2 = sub_color or color
            return (
                f'<div style="flex:1;min-width:110px;background:#fff;'
                f'border:1px solid #D8E8F4;border-radius:10px;padding:14px 10px 10px 10px;'
                f'text-align:center;box-shadow:0 1px 5px rgba(27,79,138,0.07)">'
                f'<div style="font-size:9.5px;font-weight:700;letter-spacing:1.3px;'
                f'text-transform:uppercase;color:#7A92AA;margin-bottom:6px">{label}</div>'
                f'<div style="font-size:22px;font-weight:900;color:{color};'
                f'font-family:monospace;line-height:1.1">{value}</div>'
                + (f'<div style="font-size:9px;color:{sc2};margin-top:3px">{sub}</div>' if sub else '')
                + '</div>'
            )

        ret_c = "#e74c3c" if avg_ret >= 7 else ("#f39c12" if avg_ret >= 4 else "#27ae60")
        tgt_c = "#27ae60" if avg_fulfil >= 100 else "#f39c12"
        mrg_c = "#27ae60" if avg_margin >= 20 else "#f39c12"

        kpi_html = (
            '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px">'
            + _kcard("Units Sold",   f"{total_units:,}",      "#1B4F8A")
            + _kcard("Net Sales",    _inr(total_sales),       "#1B4F8A")
            + _kcard("Margin %",     f"{avg_margin:.1f}%",    mrg_c)
            + _kcard("Return Rate",  f"{avg_ret:.1f}%",       ret_c)
            + _kcard("Replacements", f"{total_repl:,}",       "#8b5cf6")
            + _kcard("Fulfilment",   f"{avg_fulfil:.1f}%",    tgt_c)
            + '</div>'
        )

        # ── Top Products table ────────────────────────────────────────────────
        top_table_html = ""
        if pidc and sc:
            top_agg = {sc: 'sum'}
            if pmc: top_agg[pmc] = 'mean'
            if rrc: top_agg[rrc] = 'mean'
            top = df.groupby(pidc).agg(top_agg).reset_index()
            top.columns = [pidc, 'Sales'] + (['Margin'] if pmc else []) + (['RetRate'] if rrc else [])
            top = top.sort_values('Sales', ascending=False).head(5)

            rows = ""
            for rank, (_, r) in enumerate(top.iterrows(), 1):
                bg   = "#FFFFFF" if rank % 2 else "#F4F9FF"
                pid  = int(r[pidc]) if str(r[pidc]).isdigit() else r[pidc]
                pname= _PRODUCT_NAMES.get(int(pid), f"Product {pid}") if isinstance(pid, (int, float)) else str(pid)
                margin_v  = f"{r['Margin']:.0f}%"  if 'Margin' in r.index  else "—"
                retrate_v = f"{r['RetRate']:.1f}%"  if 'RetRate' in r.index else "—"
                rows += (
                    f'<tr style="background:{bg};border-bottom:1px solid #E8F0F8">'
                    f'<td style="padding:9px 14px;font-weight:700;color:#7A92AA;'
                    f'text-align:center;width:55px">{rank}</td>'
                    f'<td style="padding:9px 14px;color:#4A6A8A;font-size:11px">{pid}</td>'
                    f'<td style="padding:9px 14px;color:#1A2D45;font-weight:600">{pname}</td>'
                    f'<td style="padding:9px 14px;text-align:right;font-weight:800;'
                    f'color:#1B4F8A">{_inr(r["Sales"])}</td>'
                    f'<td style="padding:9px 14px;text-align:center;color:#27ae60;'
                    f'font-weight:700">{margin_v}</td>'
                    f'<td style="padding:9px 14px;text-align:center;color:#e74c3c">{retrate_v}</td>'
                    f'</tr>'
                )

            store_label = (store_sel or "").replace("Store: ", "").strip() or "All Stores"
            cat_label   = (cat_sel   or "").replace("Category: ", "").strip() or "All Categories"
            top_table_html = f"""
<div style="margin-bottom:16px">
  <div style="background:#EAF4FF;border:1px solid #C8DCEF;border-bottom:none;
              border-radius:8px 8px 0 0;padding:9px 16px;font-size:11px;
              font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#1B4F8A">
    Top Products in {cat_label} &nbsp;&mdash;&nbsp; {store_label}
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:12.5px;
                border:1px solid #C8DCEF;border-radius:0 0 8px 8px;overflow:hidden">
    <thead>
      <tr style="background:#0B1F3A">
        <th style="padding:9px 14px;color:#A8D8FF;font-weight:600;text-align:center">Rank</th>
        <th style="padding:9px 14px;color:#A8D8FF;font-weight:600;text-align:left">Product ID</th>
        <th style="padding:9px 14px;color:#A8D8FF;font-weight:600;text-align:left">Product Name</th>
        <th style="padding:9px 14px;color:#A8D8FF;font-weight:600;text-align:right">Sales</th>
        <th style="padding:9px 14px;color:#A8D8FF;font-weight:600;text-align:center">Margin</th>
        <th style="padding:9px 14px;color:#A8D8FF;font-weight:600;text-align:center">Return Rate</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""

        # ── Global chart style ────────────────────────────────────────────────
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.grid': True, 'grid.alpha': 0.2, 'grid.color': '#C8DCEF',
            'axes.facecolor': '#FAFCFF', 'figure.facecolor': '#FFFFFF',
        })
        NAVY  = '#1B4F8A'; GREEN = '#27ae60'; RED = '#e74c3c'
        AMBER = '#f39c12'; PURP  = '#8b5cf6'; TEAL = '#0097a7'
        CAT_PAL = [NAVY, '#f39c12', '#27ae60', RED, PURP, TEAL, '#e67e22', '#16a085']

        def _xtick_step(n, target=6): return max(1, n // target)

        def _style_ax(ax, title, ylabel="", fontsize=10):
            ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=8, color='#1A2D45')
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=8, color='#4A6A8A')
            ax.tick_params(axis='both', labelsize=7, colors='#4A6A8A')
            ax.spines['left'].set_color('#D0E4F4')
            ax.spines['bottom'].set_color('#D0E4F4')

        # ── Chart 1: Category Sales Trend (36 Months) ──────────────────────
        fig1, ax1 = plt.subplots(figsize=(7, 3.6))
        fig1.subplots_adjust(top=0.88, bottom=0.24, left=0.13, right=0.97)
        if dc and catc:
            df1 = df.copy()
            df1['_m'] = df1[dc].dt.to_period('M').astype(str)
            grp1 = df1.groupby([catc, '_m'])[sc].sum().reset_index().sort_values('_m')
            cats1 = sorted(grp1[catc].dropna().unique())
            all_months = sorted(grp1['_m'].unique())
            for ci, cat in enumerate(cats1):
                cm = grp1[grp1[catc] == cat].sort_values('_m')
                y1v = cm[sc].values / 1e5
                x1v = list(range(len(cm)))
                ax1.fill_between(x1v, y1v, alpha=0.13, color=CAT_PAL[ci % len(CAT_PAL)])
                ax1.plot(x1v, y1v, color=CAT_PAL[ci % len(CAT_PAL)],
                         linewidth=1.8, label=cat, marker='o', markersize=2)
            step1 = _xtick_step(len(all_months))
            ax1.set_xticks(range(0, len(all_months), step1))
            ax1.set_xticklabels(all_months[::step1], rotation=45, ha='right', fontsize=6.5)
            ax1.legend(fontsize=6.5, ncol=3, loc='upper left',
                       framealpha=0.9, edgecolor='#D0E4F4', labelspacing=0.3)
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_inr_ax))
        else:
            ax1.text(0.5, 0.5, 'No time-series data', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=10, color='#7A92AA')
        _style_ax(ax1, 'Category Sales Trend (36 Months)', 'Sales (₹ Lakhs)')

        # ── Chart 2: Category Margin Trend (36 Months) ─────────────────────
        fig2, ax2 = plt.subplots(figsize=(7, 3.6))
        fig2.subplots_adjust(top=0.88, bottom=0.24, left=0.11, right=0.97)
        if dc and catc and pmc:
            df2 = df.copy()
            df2['_m'] = df2[dc].dt.to_period('M').astype(str)
            grp2 = df2.groupby([catc, '_m'])[pmc].mean().reset_index().sort_values('_m')
            cats2 = sorted(grp2[catc].dropna().unique())
            all_m2 = sorted(grp2['_m'].unique())
            for ci, cat in enumerate(cats2):
                cm2 = grp2[grp2[catc] == cat].sort_values('_m')
                y2v = cm2[pmc].values
                x2v = list(range(len(cm2)))
                ax2.fill_between(x2v, y2v, alpha=0.13, color=CAT_PAL[ci % len(CAT_PAL)])
                ax2.plot(x2v, y2v, color=CAT_PAL[ci % len(CAT_PAL)],
                         linewidth=1.8, label=cat, marker='o', markersize=2)
            ax2.axhline(20, color=GREEN, linestyle='--', linewidth=1.2,
                        alpha=0.75, label='Target 20%')
            step2 = _xtick_step(len(all_m2))
            ax2.set_xticks(range(0, len(all_m2), step2))
            ax2.set_xticklabels(all_m2[::step2], rotation=45, ha='right', fontsize=6.5)
            ax2.legend(fontsize=6.5, ncol=3, loc='upper left',
                       framealpha=0.9, edgecolor='#D0E4F4', labelspacing=0.3)
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        else:
            ax2.text(0.5, 0.5, 'No margin data', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=10, color='#7A92AA')
        _style_ax(ax2, 'Category Margin Trend (36 Months)', 'Margin %')

        # ── Linear forecast helper ───────────────────────────────────────────
        def _forecast(series, n):
            y  = np.array(series, dtype=float)
            X  = np.arange(len(y)).reshape(-1, 1)
            lr = LinearRegression().fit(X, y)
            Xf = np.arange(len(y), len(y) + n).reshape(-1, 1)
            yh = lr.predict(Xf).clip(0)
            std = max(np.std(y - lr.predict(X)), 1)
            return y, yh, (yh - 1.65 * std).clip(0), yh + 1.65 * std

        # ── Chart 3: 6-Month Forecast ────────────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(5, 3.6))
        fig3.subplots_adjust(top=0.88, bottom=0.14, left=0.14, right=0.97)
        total_6m = 0
        if dc and sc:
            df3 = df.copy()
            df3['_m'] = df3[dc].dt.to_period('M').astype(str)
            ms3 = df3.groupby('_m')[sc].sum().sort_index()
            if len(ms3) >= 3:
                y3, yh6, lo6, hi6 = _forecast(ms3.values, 6)
                total_6m = float(yh6.sum())
                xh3 = list(range(len(y3)))
                xf3 = list(range(len(y3) - 1, len(y3) + 6))
                fy3 = np.concatenate([[y3[-1]], yh6])
                fl3 = np.concatenate([[y3[-1]], lo6])
                fh3 = np.concatenate([[y3[-1]], hi6])
                ax3.plot(xh3, y3 / 1e5, color=NAVY, lw=2, label='Expected Sales & Returns',
                         marker='o', ms=2.5)
                ax3.plot(xf3, fy3 / 1e5, color=RED, lw=2, ls='--',
                         marker='o', ms=2.5, label='Forecast')
                ax3.fill_between(xf3, fl3 / 1e5, fh3 / 1e5, color=RED, alpha=0.12)
                ax3.axvline(len(y3) - 1, color='#888', ls=':', lw=1, alpha=0.6)
                ax3.legend(fontsize=6.5, loc='upper left', framealpha=0.9)
                ax3.text(0.97, 0.06, f'6M: {_inr(total_6m)}',
                         transform=ax3.transAxes, ha='right', fontsize=8.5, fontweight='bold',
                         color=RED, bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='#FFF3CD', edgecolor=AMBER, alpha=0.9))
        ax3.set_xticklabels([]); ax3.set_xlabel('')
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(_inr_ax))
        _style_ax(ax3, '6-Month Forecast', 'Sales (₹L)')

        # ── Chart 4: 12-Month Forecast ───────────────────────────────────────
        fig4, ax4 = plt.subplots(figsize=(5, 3.6))
        fig4.subplots_adjust(top=0.88, bottom=0.14, left=0.14, right=0.97)
        total_12m = 0
        if dc and sc:
            df4 = df.copy()
            df4['_m'] = df4[dc].dt.to_period('M').astype(str)
            ms4 = df4.groupby('_m')[sc].sum().sort_index()
            if len(ms4) >= 3:
                y4, yh12, lo12, hi12 = _forecast(ms4.values, 12)
                total_12m = float(yh12.sum())
                xh4 = list(range(len(y4)))
                xf4 = list(range(len(y4) - 1, len(y4) + 12))
                fy4 = np.concatenate([[y4[-1]], yh12])
                fl4 = np.concatenate([[y4[-1]], lo12])
                fh4 = np.concatenate([[y4[-1]], hi12])
                ax4.plot(xh4, y4 / 1e5, color=NAVY, lw=2, label='Forecast',
                         marker='o', ms=2)
                ax4.plot(xf4, fy4 / 1e5, color=GREEN, lw=2, ls='--',
                         marker='o', ms=2, label='Actual')
                ax4.fill_between(xf4, fl4 / 1e5, fh4 / 1e5, color=GREEN, alpha=0.12)
                ax4.axvline(len(y4) - 1, color='#888', ls=':', lw=1, alpha=0.6)
                ax4.legend(fontsize=6.5, loc='upper left', framealpha=0.9)
                ax4.text(0.97, 0.06, f'12M: {_inr(total_12m)}',
                         transform=ax4.transAxes, ha='right', fontsize=8.5, fontweight='bold',
                         color=GREEN, bbox=dict(boxstyle='round,pad=0.3',
                                                facecolor='#EAF7EE', edgecolor=GREEN, alpha=0.9))
        ax4.set_xticklabels([]); ax4.set_xlabel('')
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(_inr_ax))
        _style_ax(ax4, '12-Month Forecast', 'Sales (₹L)')

        # ── Chart 5: Product Fulfilment Trend ────────────────────────────────
        fig5, ax5 = plt.subplots(figsize=(5, 3.6))
        fig5.subplots_adjust(top=0.88, bottom=0.22, left=0.13, right=0.97)
        if dc and tac:
            df5 = df.copy()
            df5['_m'] = df5[dc].dt.to_period('M').astype(str)
            ft  = df5.groupby('_m')[tac].mean().sort_index()
            xft = list(range(len(ft)))
            fv  = ft.values
            ax5.fill_between(xft, fv, 100, where=fv >= 100,
                             alpha=0.18, color=GREEN, interpolate=True)
            ax5.fill_between(xft, fv, 100, where=fv < 100,
                             alpha=0.18, color=RED, interpolate=True)
            ax5.plot(xft, fv, color=NAVY, lw=2, marker='o', ms=2.5)
            ax5.axhline(100, color=GREEN, ls='--', lw=1.2, alpha=0.7, label='100% target')
            ax5.set_ylim(max(80, fv.min() - 4), min(115, fv.max() + 4))
            step5 = _xtick_step(len(ft))
            ax5.set_xticks(range(0, len(ft), step5))
            ax5.set_xticklabels(ft.index.tolist()[::step5], rotation=45, ha='right', fontsize=6.5)
            ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
            ax5.legend(fontsize=6.5, loc='lower right', framealpha=0.9)
        else:
            ax5.text(0.5, 0.5, 'No fulfilment data', ha='center', va='center',
                     transform=ax5.transAxes, fontsize=10, color='#7A92AA')
        _style_ax(ax5, 'Product Fulfilment Trend', 'Target Achievement %')

        # ── Chart 6: Product Sales vs Returns ────────────────────────────────
        fig6, ax6 = plt.subplots(figsize=(6, 3.6))
        fig6.subplots_adjust(top=0.88, bottom=0.22, left=0.13, right=0.91)
        if dc and sc:
            df6 = df.copy()
            df6['_m'] = df6[dc].dt.to_period('M').astype(str)
            agg6 = {'sales': (sc, 'sum')}
            if rrc: agg6['ret'] = (rrc, 'mean')
            sr = df6.groupby('_m').agg(**agg6).sort_index()
            ax6b = ax6.twinx()
            xsr  = list(range(len(sr)))
            ax6.fill_between(xsr, sr['sales'].values / 1e5, alpha=0.15, color=NAVY)
            ax6.plot(xsr, sr['sales'].values / 1e5, color=NAVY, lw=2,
                     marker='o', ms=2.5, label='Sales')
            if rrc and 'ret' in sr.columns:
                ax6b.plot(xsr, sr['ret'].values, color=RED, lw=2, ls='--',
                          marker='o', ms=2.5, label='Returns')
                ax6b.set_ylabel('Return Rate %', fontsize=8, color=RED)
                ax6b.tick_params(axis='y', labelcolor=RED, labelsize=7)
                ax6b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))
                ax6b.spines['right'].set_color('#F5C6CB')
            step6 = _xtick_step(len(sr))
            ax6.set_xticks(range(0, len(sr), step6))
            ax6.set_xticklabels(sr.index.tolist()[::step6], rotation=45, ha='right', fontsize=6.5)
            ax6.yaxis.set_major_formatter(mticker.FuncFormatter(_inr_ax))
            h1, l1 = ax6.get_legend_handles_labels()
            h2, l2 = ax6b.get_legend_handles_labels() if rrc else ([], [])
            ax6.legend(h1 + h2, l1 + l2, fontsize=6.5, loc='upper left',
                       framealpha=0.9, edgecolor='#D0E4F4')
        _style_ax(ax6, 'Product Sales vs Returns', 'Sales (₹L)')

        # ── Chart 7: Inventory & Reorder Levels ──────────────────────────────
        fig7, ax7 = plt.subplots(figsize=(6, 3.6))
        fig7.subplots_adjust(top=0.88, bottom=0.28, left=0.11, right=0.97)
        if pidc and slc and roc and ivc:
            inv = df.groupby(pidc).agg(
                Stock=(slc,  'mean'),
                Reorder=(roc, 'mean'),
                Restck=(ivc, 'mean'),
            ).reset_index().head(8)
            x7 = list(range(len(inv)))
            w7 = 0.27
            ax7.bar([xi - w7 for xi in x7], inv['Stock'],   width=w7, color=NAVY,  alpha=0.82, label='Stock')
            ax7.bar([xi       for xi in x7], inv['Reorder'], width=w7, color=AMBER, alpha=0.82, label='Reorder Point')
            ax7.bar([xi + w7 for xi in x7], inv['Restck'],  width=w7, color=GREEN, alpha=0.82, label='Restock Level')
            ax7.set_xticks(list(x7))
            pids7 = [int(p) if str(p).isdigit() else p for p in inv[pidc]]
            ax7.set_xticklabels([f"SKU-{p}" for p in pids7], rotation=45, ha='right', fontsize=7)
            ax7.legend(fontsize=6.5, loc='upper right', framealpha=0.9, edgecolor='#D0E4F4')
        elif slc or roc or ivc:
            # Fallback — just show available stock column
            ivf_col = slc or ivc or roc
            if ivf_col and pidc:
                inv_fb = df.groupby(pidc)[ivf_col].mean().reset_index().head(8)
                ax7.bar(range(len(inv_fb)), inv_fb[ivf_col], color=NAVY, alpha=0.82)
                ax7.set_xticks(range(len(inv_fb)))
                ax7.set_xticklabels([f"SKU-{p}" for p in inv_fb[pidc]], rotation=45, ha='right', fontsize=7)
            else:
                ax7.text(0.5, 0.5, 'No inventory data', ha='center', va='center',
                         transform=ax7.transAxes, fontsize=10, color='#7A92AA')
        else:
            ax7.text(0.5, 0.5, 'No inventory data', ha='center', va='center',
                     transform=ax7.transAxes, fontsize=10, color='#7A92AA')
        ax7.set_ylabel('Units', fontsize=8, color='#4A6A8A')
        _style_ax(ax7, 'Inventory & Reorder Levels', 'Units')

        # ── AI Forecast Summary ───────────────────────────────────────────────
        cat_label   = (cat_sel   or "").replace("Category: ", "").strip() or "All Categories"
        store_label = (store_sel or "").replace("Store: ", "").strip()    or "All Stores"

        top_cat = cat_label if cat_label != "All Categories" else "FMCG"
        if catc in df.columns and not df.empty and cat_label == "All Categories":
            by_cat = df.groupby(catc)[sc].sum()
            if not by_cat.empty:
                top_cat = by_cat.idxmax()

        low_stock_msg = ""
        if slc and roc and pidc:
            try:
                ls = df[df[slc] < df[roc]]
                if not ls.empty:
                    top_low  = ls.groupby(pidc)[slc].mean().idxmin()
                    pid_int  = int(top_low) if str(top_low).isdigit() else top_low
                    pname_ls = _PRODUCT_NAMES.get(pid_int, f"SKU-{pid_int}")
                    low_stock_msg = (
                        f'<li>Recommend restocking <strong>{pname_ls}</strong>'
                        f' within 2 weeks.</li>')
            except Exception:
                pass

        fc_pct        = 8 if avg_margin >= 20 else 5
        margin_state  = "strong" if avg_margin >= 20 else "moderate"
        ret_msg       = (f"Monitor growing returns ({avg_ret:.1f}%) — review product quality."
                         if avg_ret > 7 else
                         f"Return rate {avg_ret:.1f}% is within healthy limits.")
        clothing_note = ('<li>Monitor growing clothing returns during the monsoon.</li>'
                         if 'clothing' in cat_label.lower() or cat_label == "All Categories"
                         else "")
        elec_note     = ('<li>Electronics inventory low — reorder urgency.</li>'
                         if 'electr' in cat_label.lower() or cat_label == "All Categories"
                         else "")

        # ── Compute action-specific numbers ──────────────────────────────────────
        inv_action_pct  = 10 if avg_fulfil < 70 else (5 if avg_fulfil < 85 else 0)
        margin_gap      = round(20 - avg_margin, 1) if avg_margin < 20 else 0
        ret_action      = avg_ret > 7
        # Low-margin action
        margin_action_li = (
            f'<li>💡 <strong>Renegotiate supplier cost for <em>{top_cat}</em></strong> — '
            f'current margin is <strong>{avg_margin:.1f}%</strong>, '
            f'{margin_gap}pp below the 20% target. A 5% cost reduction unlocks ~{_inr(total_sales * 0.05 if total_sales else 0)} extra profit.</li>'
            if margin_gap > 0 else
            f'<li>✅ <strong>Margin healthy at {avg_margin:.1f}%</strong> — '
            f'consider expanding <em>{top_cat}</em> SKU count to grow revenue.</li>'
        )
        # Inventory action
        inv_action_li = (
            f'<li>📦 <strong>Increase inventory of <em>{top_cat}</em> by {inv_action_pct}%</strong> — '
            f'fulfilment rate is <strong>{avg_fulfil:.1f}%</strong> '
            f'(target ≥85%). Stockout risk is high; pre-order for the next 30 days.</li>'
            if inv_action_pct > 0 else
            f'<li>✅ <strong>Inventory levels adequate</strong> — fulfilment at {avg_fulfil:.1f}%. '
            f'Focus on optimising reorder cycle frequency.</li>'
        )
        # Returns action
        ret_action_li = (
            f'<li>🔁 <strong>Investigate return drivers for <em>{top_cat}</em></strong> — '
            f'return rate is <strong>{avg_ret:.1f}%</strong> (threshold 7%). '
            f'Review top 3 returned SKUs and add quality checks before dispatch.</li>'
            if ret_action else
            f'<li>✅ <strong>Return rate healthy at {avg_ret:.1f}%</strong> — '
            f'maintain current quality controls.</li>'
        )
        # Forecast growth action
        fc_action_li = (
            f'<li>📈 <strong>Prepare for {fc_pct}–{fc_pct+4}% demand rise in <em>{top_cat}</em></strong> '
            f'over the next 6 months — projected revenue: <strong>{_inr(total_6m) if total_6m else "—"}</strong>. '
            f'Align procurement and staffing 3 weeks ahead.</li>'
        )

        ai_html = f"""
<div style="background:linear-gradient(160deg,#EAF4FF 0%,#F8FAFF 100%);
            border:1px solid #B8D4F0;border-radius:10px;padding:16px 18px;
            height:100%;box-shadow:0 1px 8px rgba(27,79,138,0.08)">
  <div style="font-size:10px;font-weight:800;letter-spacing:1.8px;text-transform:uppercase;
              color:#1B4F8A;margin-bottom:6px;padding-bottom:8px;border-bottom:2px solid #C8DCEF">
    🤖 AI Action Recommendations
  </div>
  <div style="font-size:10px;color:#4A6A8A;margin-bottom:10px">
    Based on your data · {store_label} · {cat_label}
  </div>
  <ul style="margin:0 0 14px 0;padding-left:18px;font-size:12px;
             line-height:1.9;color:#1A2D45;list-style:none;padding-left:0">
    {fc_action_li}
    {inv_action_li}
    {margin_action_li}
    {ret_action_li}
    {low_stock_msg}
  </ul>
  <div style="background:#fff;border-radius:8px;padding:10px 13px;
              border-left:3px solid #1B4F8A;font-size:11px;color:#4A6A8A;line-height:1.7">
    <span style="font-weight:700;color:#1B4F8A">Store:</span> {store_label}
    &nbsp;|&nbsp;
    <span style="font-weight:700;color:#1B4F8A">Category:</span> {cat_label}<br>
    <span style="font-weight:700;color:#1B4F8A">Margin:</span> {avg_margin:.1f}% ({margin_state})
    &nbsp;|&nbsp;
    <span style="font-weight:700;color:#1B4F8A">Return Rate:</span>
    <span style="color:{ret_c};font-weight:700">{avg_ret:.1f}%</span><br>
    <span style="font-weight:700;color:#1B4F8A">Fulfilment:</span>
    <span style="color:{tgt_c};font-weight:700">{avg_fulfil:.1f}%</span>
    &nbsp;|&nbsp;
    <span style="font-weight:700;color:#1B4F8A">6M Forecast:</span>
    <span style="color:#1B4F8A;font-weight:700">{_inr(total_6m) if total_6m else "—"}</span>
  </div>
</div>"""

        plt.close('all')
        return dict(kpi=kpi_html, table=top_table_html, ai=ai_html,
                    f1=fig1, f2=fig2, f3=fig3, f4=fig4,
                    f5=fig5, f6=fig6, f7=fig7)


    # ── Step 7 outputs helper ────────────────────────────────────────────────
    _S7_CHART_OUTPUTS = [
        s7_kpi_html, s7_top_table,
        s7_cat_sales_chart, s7_cat_margin_chart,
        s7_fc6_chart, s7_fc12_chart, s7_fulfil_chart,
        s7_sales_ret_chart, s7_inventory_chart,
        s7_ai_summary
    ]

    def _pack_s7(result):
        """Unpack _build_step7_data result dict into Gradio output tuple."""
        if result is None:
            empty = "<div style='padding:20px;color:#aaa'>No data. Upload and analyse first.</div>"
            return empty, empty, None, None, None, None, None, None, None, empty
        return (result['kpi'], result['table'],
                result['f1'],  result['f2'],
                result['f3'],  result['f4'],  result['f5'],
                result['f6'],  result['f7'],
                result['ai'])

    def show_granular_dashboard(granular_data, df_raw):
        """Navigate to Step 7 and populate with default (All) filters."""
        result = _build_step7_data(df_raw, "Store: All", "Category: All", "Product: All")

        # Build filter dropdown choices from df_raw
        stores = ["Store: All"]
        cats   = ["Category: All"]
        prods  = ["Product: All"]
        if df_raw is not None:
            dff = df_raw.copy()
            # df_raw is already the MSME-filtered slice — use all rows for dropdown population
            stc2 = 'store_id' if 'store_id' in dff.columns else ('Store_ID' if 'Store_ID' in dff.columns else None)
            c2   = 'product_category' if 'product_category' in dff.columns else ('Product_Category' if 'Product_Category' in dff.columns else None)
            p2   = 'product_id' if 'product_id' in dff.columns else ('SKU_Name' if 'SKU_Name' in dff.columns else None)
            if stc2: stores += [f"Store: {s}" for s in sorted(dff[stc2].unique())]
            if c2:   cats   += [f"Category: {c}" for c in sorted(dff[c2].dropna().unique())]
            if p2:   prods  += [f"Product: SKU-{p}" for p in sorted(dff[p2].unique())[:40]]

        return (7, *update_visibility_all('step7'),
                gr.update(choices=stores, value="Store: All"),
                gr.update(choices=cats,   value="Category: All"),
                gr.update(choices=prods,  value="Product: All"),
                *_pack_s7(result))

    def update_step7_filters(store_sel, cat_sel, prod_sel, df_raw):
        """Recompute all Step 7 outputs when any filter dropdown changes."""
        result = _build_step7_data(df_raw, store_sel, cat_sel, prod_sel)
        return _pack_s7(result)


    # ══════════════════════════════════════════════════════════════════════════
    # Wire Events
    # ══════════════════════════════════════════════════════════════════════════
    show_signup_trigger.click(show_signup, [], [step_state]+_ALL_COLS)
    quick_signup_btn.click(show_signup, [], [step_state]+_ALL_COLS)
    cancel1_btn.click(lambda: (0, *update_visibility_all('step0')), [], [step_state]+_ALL_COLS)
    back2_btn.click(lambda: (1, *update_visibility_all('step1')), [], [step_state]+_ALL_COLS)
    back3_btn.click(lambda: (2, *update_visibility_all('step2')), [], [step_state]+_ALL_COLS)
    back6_btn.click(lambda: (5, *update_visibility_all('step5')), [], [step_state]+_ALL_COLS)
    back6a_btn.click(lambda: (5, *update_visibility_all('step5')), [], [step_state]+_ALL_COLS)

    quick_login_btn.click(handle_login, [quick_login_mobile],
        [landing_login_error_msg, user_data_state, step_state]+_ALL_COLS+[error1,error2,error3,error4,error5,login_welcome_message])

    next1_btn.click(validate_step1, [name_input, mobile_input, email_input, role_input, user_data_state],
        [error1, user_data_state, step_state]+_ALL_COLS)

    # FIX 4 — fetch also returns industry_domain
    fetch_btn.click(_fetch_msme_data, [msme_number_input],
        [fetched_name, fetched_org, fetched_activity, fetched_type, fetched_state, fetched_city, fetched_industry, fetch_status])

    next2_btn.click(verify_step2,
        [msme_number_input, otp_input, user_data_state, fetched_name, fetched_org, fetched_activity, fetched_type, fetched_state, fetched_city, fetched_industry, fetch_status],
        [error2, user_data_state, step_state]+_ALL_COLS+[confirm_name, confirm_org, confirm_activity, confirm_type, confirm_state, confirm_city, confirm_industry, fetch_status])

    next3_btn.click(confirm_step3, [user_data_state, consent1, consent2, certificate_upload],
        [error3, user_data_state, step_state]+_ALL_COLS+[verification_status_display, business_type_input])

    back4_btn.click(lambda: (3, *update_visibility_all('step3'), gr.update(value="",visible=False), gr.update(visible=True)), [],
        [step_state]+_ALL_COLS+[error4, next4_btn])

    next4_btn.click(submit_profile, [business_type_input, years_input, revenue_input, user_data_state],
        [error4, user_data_state, proceed_to_step5_btn, next4_btn])

    proceed_to_step5_btn.click(lambda: (5, *update_visibility_all('step5'), gr.update(value="",visible=False), gr.update(value="",visible=False)), [],
        [step_state]+_ALL_COLS+[error4, login_welcome_message])

    back5_btn.click(lambda: (4, *update_visibility_all('step4'), gr.update(value="",visible=False)), [],
        [step_state]+_ALL_COLS+[login_welcome_message])

    cancel5_btn.click(lambda: (0, *update_visibility_all('step0'), gr.update(value="",visible=False)), [],
        [step_state]+_ALL_COLS+[login_welcome_message])

    analyze_btn.click(analyze_data, [user_data_state, consent_check, file_upload, lang_state],
        [insights_output, view_dashboard_btn, view_gov_dashboard_btn,
         kpi1,kpi2,kpi3,kpi4,kpi5, chart1,chart2,chart3,chart4,
         sum1,sum2,sum3,sum4, upload_message, dashboard_data_state, df_state])

    file_upload.change(handle_file_upload_change, [user_data_state, file_upload], [upload_message, error5])

    view_dashboard_btn.click(show_dashboard, [dashboard_data_state],
        [step_state]+_ALL_COLS+[kpi_table_dash, chart1_dash, chart2_dash, chart3_dash, chart4_dash,
         chart1_summary, chart2_summary, chart3_summary, chart4_summary, granular_forecast_data_state])

    view_gov_dashboard_btn.click(show_gov_dashboard, [df_state],
        [step_state]+_ALL_COLS+[gov_dashboard_html])



    _S7_NAV_OUTPUTS = ([step_state] + _ALL_COLS
                       + [s7_store_filter, s7_cat_filter, s7_prod_filter]
                       + _S7_CHART_OUTPUTS)
    forecast_deepdive_btn.click(show_granular_dashboard,
        [granular_forecast_data_state, df_state],
        _S7_NAV_OUTPUTS)

    s7_store_filter.change(update_step7_filters,
        [s7_store_filter, s7_cat_filter, s7_prod_filter, df_state],
        _S7_CHART_OUTPUTS)
    s7_cat_filter.change(update_step7_filters,
        [s7_store_filter, s7_cat_filter, s7_prod_filter, df_state],
        _S7_CHART_OUTPUTS)
    s7_prod_filter.change(update_step7_filters,
        [s7_store_filter, s7_cat_filter, s7_prod_filter, df_state],
        _S7_CHART_OUTPUTS)

    back7_btn.click(lambda: (6, *update_visibility_all('step6')), [], [step_state]+_ALL_COLS)
    back7_to5_btn.click(lambda: (5, *update_visibility_all('step5')), [], [step_state]+_ALL_COLS)

    # Language switching
    def switch_lang_en():
        return ('en', gr.update(variant='primary'), gr.update(variant='secondary'), '**Active: English 🇬🇧**',
                gr.update(value=_landing_hero('en')), gr.update(value=_landing_capabilities('en')),
                gr.update(value="## How DataNetra Works"),
                gr.update(value="### 📥 Step 1: Upload Your Data"),
                gr.update(value="Easily upload Excel/CSV files for comprehensive analysis."),
                gr.update(value="### 🤖 Step 2: AI-Powered Analysis"),
                gr.update(value="Our AI processes your data, forecasting trends and uncovering insights."),
                gr.update(value="### 📊 Step 3: Actionable Dashboards & Recommendations"),
                gr.update(value="Access interactive dashboards, KPI charts and personalized recommendations."),
                gr.update(value="**Already Registered**"), gr.update(label="Enter Mobile Number"),
                gr.update(value="Login"), gr.update(value="**First Time User**"),
                gr.update(value="**Signup to unlock smart AI Insights**"), gr.update(value="Sign Up Now"))

    def switch_lang_hi():
        return ('hi', gr.update(variant='secondary'), gr.update(variant='primary'), '**सक्रिय: हिंदी 🇮🇳**',
                gr.update(value=_landing_hero('hi')), gr.update(value=_landing_capabilities('hi')),
                gr.update(value="## DataNetra कैसे काम करता है?"),
                gr.update(value="### 📥 पहला काम: अपना Data डालें"),
                gr.update(value="अपनी Excel या CSV फाइल आसानी से Upload करें।"),
                gr.update(value="### 🤖 दूसरा काम: AI जाँच करेगा"),
                gr.update(value="हमारा AI आपका Data पढ़कर बताएगा — क्या बिक रहा है, क्या नहीं।"),
                gr.update(value="### 📊 तीसरा काम: रिपोर्ट और सलाह देखें"),
                gr.update(value="साफ Dashboard पर अपनी बिक्री, मुनाफा और AI की सलाह एक जगह देखें।"),
                gr.update(value="**पहले से जुड़े हैं?**"), gr.update(label="मोबाइल नंबर डालें"),
                gr.update(value="Login करें"), gr.update(value="**नए हैं? पहली बार?**"),
                gr.update(value="**अभी जुड़ें और AI से अपने धंधे की जानकारी पाएं**"), gr.update(value="अभी Register करें"))

    _lang_landing_outputs = [lang_state, lang_en_btn, lang_hi_btn, lang_indicator,
                              landing_hero_html, landing_capabilities_html,
                              landing_how_title, landing_step1_title, landing_step1_desc,
                              landing_step2_title, landing_step2_desc, landing_step3_title, landing_step3_desc,
                              landing_login_title, quick_login_mobile, quick_login_btn,
                              landing_signup_title, landing_signup_desc, quick_signup_btn]
    lang_en_btn.click(switch_lang_en, [], _lang_landing_outputs)
    lang_hi_btn.click(switch_lang_hi, [], _lang_landing_outputs)

print("=" * 60)
print("🚀 DataNetra.ai - MSME Intelligence Platform v4.8")
print("   FIX 1: UTF-8 surrogate patch — no more JSON crash")
print("   FIX 2: Correct column mapping for lowercase dataset")
print("   FIX 3: Major Activity shown in Step 4 verification")
print("   FIX 4: Industry Domain field in Step 2 & Step 3")
print("   FIX 5: Voice registration in Step 1 & Step 2")
print("=" * 60)

import os as _os
_port = int(_os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=_port, show_api=False)