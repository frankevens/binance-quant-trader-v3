"""
Binance Quant Trader V3 - Web Dashboard
Real-time monitoring panel for trading bot
"""
import os
import sys
import json
import sqlite3
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request

from config import config as default_config

app = Flask(__name__, template_folder='templates', static_folder='static')

DB_PATH = default_config.db_path


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Get overall bot status"""
    try:
        conn = get_db()
        
        # Today's stats
        today = datetime.utcnow().strftime('%Y-%m-%d')
        today_trades = conn.execute(
            "SELECT COUNT(*) as cnt, SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins "
            "FROM trades WHERE DATE(opened_at) = ?", (today,)
        ).fetchone()
        
        today_pnl = conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) as pnl FROM trades WHERE DATE(opened_at) = ?",
            (today,)
        ).fetchone()
        
        # Total stats
        total = conn.execute(
            "SELECT COUNT(*) as cnt, "
            "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
            "COALESCE(SUM(realized_pnl), 0) as total_pnl, "
            "COALESCE(AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END), 0) as avg_win, "
            "COALESCE(AVG(CASE WHEN realized_pnl <= 0 THEN realized_pnl END), 0) as avg_loss "
            "FROM trades WHERE status IN ('closed', 'stopped')"
        ).fetchone()
        
        # Active positions
        positions = conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(unrealized_pnl), 0) as upnl "
            "FROM positions WHERE status = 'open'"
        ).fetchone()
        
        # Recent events
        events = conn.execute(
            "SELECT type, message, created_at FROM events ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        
        total_cnt = total['cnt'] or 0
        total_wins = total['wins'] or 0
        win_rate = (total_wins / total_cnt * 100) if total_cnt > 0 else 0
        avg_win = total['avg_win'] or 0
        avg_loss = abs(total['avg_loss']) if total['avg_loss'] else 0
        profit_factor = (avg_win / avg_loss) if avg_loss > 0 else 0
        
        conn.close()
        
        return jsonify({
            'today_trades': today_trades['cnt'] or 0,
            'today_wins': today_trades['wins'] or 0,
            'today_pnl': float(today_pnl['pnl'] or 0),
            'total_trades': total_cnt,
            'total_wins': total_wins,
            'win_rate': round(win_rate, 1),
            'total_pnl': float(total['total_pnl'] or 0),
            'profit_factor': round(profit_factor, 2),
            'avg_win': float(avg_win),
            'avg_loss': float(total['avg_loss'] or 0),
            'open_positions': positions['cnt'] or 0,
            'unrealized_pnl': float(positions['upnl'] or 0),
            'recent_events': [dict(e) for e in events],
            'bot_running': True,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e), 'bot_running': False})


@app.route('/api/positions')
def api_positions():
    """Get all positions"""
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY opened_at DESC"
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/trades')
def api_trades():
    """Get trade history"""
    limit = request.args.get('limit', 50, type=int)
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/daily_pnl')
def api_daily_pnl():
    """Get daily PnL for chart"""
    days = request.args.get('days', 30, type=int)
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT date, realized_pnl, cumulative_pnl, trades_count "
            "FROM daily_pnl ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
        conn.close()
        data = [dict(r) for r in rows]
        data.reverse()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/trade_distribution')
def api_trade_distribution():
    """Get trade PnL distribution for histogram"""
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT realized_pnl FROM trades WHERE status IN ('closed', 'stopped') "
            "ORDER BY opened_at DESC LIMIT 200"
        ).fetchall()
        conn.close()
        pnls = [float(r['realized_pnl'] or 0) for r in rows]
        return jsonify(pnls)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/symbols')
def api_symbols():
    """Get per-symbol stats"""
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT symbol, "
            "COUNT(*) as trades, "
            "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
            "COALESCE(SUM(realized_pnl), 0) as total_pnl, "
            "COALESCE(AVG(realized_pnl), 0) as avg_pnl "
            "FROM trades WHERE status IN ('closed', 'stopped') "
            "GROUP BY symbol ORDER BY total_pnl DESC"
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            d = dict(r)
            d['win_rate'] = round(d['wins'] / d['trades'] * 100, 1) if d['trades'] > 0 else 0
            result.append(d)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/config')
def api_config():
    """Get current config"""
    return jsonify({
        'symbols': default_config.symbols,
        'leverage': 10,
        'margin_type': 'ISOLATED',
        'kline_interval': default_config.atr.kline_interval,
        'atr_period': default_config.atr.atr_period,
        'min_entry_score': default_config.atr.min_entry_score,
        'sl_mult': default_config.atr.atr_sl_multiplier,
        'tp1_rr': default_config.atr.partial_tp_tp1_rr,
        'tp2_rr': default_config.atr.partial_tp_tp2_rr,
        'tp3_rr': default_config.atr.partial_tp_tp3_rr,
        'risk_per_trade': default_config.risk.max_position_pct,
        'max_daily_loss_pct': default_config.risk.max_daily_loss_pct,
        'max_total_exposure_pct': default_config.risk.max_total_exposure_pct,
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Binance Quant Trader V3 - Web Dashboard")
    print("=" * 60)
    port = int(os.environ.get('DASHBOARD_PORT', 5000))
    print(f"  URL: http://0.0.0.0:{port}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
