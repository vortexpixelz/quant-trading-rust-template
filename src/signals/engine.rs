//! Signal generation engine.
//!
//! Computes quantitative signals from orderbook data:
//! - Price Z-Score (mean reversion signal)
//! - Spread Z-Score (liquidity signal)
//! - Imbalance momentum (directional signal)
//! - VPIN approximation (toxicity signal)
//! - Kyle's lambda trend (market impact signal)

use std::collections::VecDeque;

/// Computed signals for strategy consumption.
#[derive(Debug, Clone)]
pub struct Signals {
    /// Price Z-score: how far current mid-price is from rolling mean.
    /// |z| > 2 = potential mean-reversion opportunity.
    pub price_zscore: f64,

    /// Spread Z-score: how current spread compares to rolling mean.
    /// High positive = unusually wide spread (low liquidity).
    pub spread_zscore: f64,

    /// Rolling imbalance: smoothed bid-ask imbalance.
    /// Positive = buy pressure, negative = sell pressure.
    pub imbalance_ema: f64,

    /// Imbalance momentum: rate of change of imbalance.
    /// Positive = increasing buy pressure.
    pub imbalance_momentum: f64,

    /// VPIN approximation from depth flow.
    /// Higher = more informed/toxic flow.
    pub vpin: f64,

    /// Current mid-price.
    pub mid_price: f64,

    /// Current spread.
    pub spread: f64,

    /// Current imbalance.
    pub imbalance: f64,

    /// Bid-side depth.
    pub bid_depth: f64,

    /// Ask-side depth.
    pub ask_depth: f64,
}

/// Rolling signal computation engine.
pub struct SignalEngine {
    lookback: usize,
    prices: VecDeque<f64>,
    spreads: VecDeque<f64>,
    imbalances: VecDeque<f64>,
    bid_depths: VecDeque<f64>,
    ask_depths: VecDeque<f64>,
    imbalance_ema: f64,
    ema_alpha: f64,
}

impl SignalEngine {
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            prices: VecDeque::with_capacity(lookback + 1),
            spreads: VecDeque::with_capacity(lookback + 1),
            imbalances: VecDeque::with_capacity(lookback + 1),
            bid_depths: VecDeque::with_capacity(lookback + 1),
            ask_depths: VecDeque::with_capacity(lookback + 1),
            imbalance_ema: 0.0,
            ema_alpha: 2.0 / (lookback as f64 + 1.0),
        }
    }

    /// Push a new observation into the rolling window.
    pub fn push(
        &mut self,
        mid_price: f64,
        spread: f64,
        imbalance: f64,
        bid_depth: f64,
        ask_depth: f64,
    ) {
        self.prices.push_back(mid_price);
        self.spreads.push_back(spread);
        self.imbalances.push_back(imbalance);
        self.bid_depths.push_back(bid_depth);
        self.ask_depths.push_back(ask_depth);

        // Trim to lookback window
        while self.prices.len() > self.lookback {
            self.prices.pop_front();
            self.spreads.pop_front();
            self.imbalances.pop_front();
            self.bid_depths.pop_front();
            self.ask_depths.pop_front();
        }

        // Update EMA
        self.imbalance_ema =
            self.ema_alpha * imbalance + (1.0 - self.ema_alpha) * self.imbalance_ema;
    }

    /// Compute all signals. Returns None if insufficient data.
    pub fn compute(&self) -> Option<Signals> {
        if self.prices.len() < 5 {
            return None;
        }

        let mid = *self.prices.back()?;
        let spread = *self.spreads.back()?;
        let imbalance = *self.imbalances.back()?;
        let bid_depth = *self.bid_depths.back()?;
        let ask_depth = *self.ask_depths.back()?;

        Some(Signals {
            price_zscore: self.zscore(&self.prices),
            spread_zscore: self.zscore(&self.spreads),
            imbalance_ema: self.imbalance_ema,
            imbalance_momentum: self.momentum(&self.imbalances, 5),
            vpin: self.compute_vpin(),
            mid_price: mid,
            spread,
            imbalance,
            bid_depth,
            ask_depth,
        })
    }

    /// Z-score of the latest value in a rolling window.
    fn zscore(&self, data: &VecDeque<f64>) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let n = data.len() as f64;
        let mean: f64 = data.iter().sum::<f64>() / n;
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        let latest = data.back().unwrap();
        (latest - mean) / std
    }

    /// Rate of change over last N periods.
    fn momentum(&self, data: &VecDeque<f64>, periods: usize) -> f64 {
        if data.len() < periods + 1 {
            return 0.0;
        }
        let current = data[data.len() - 1];
        let past = data[data.len() - 1 - periods];
        current - past
    }

    /// VPIN approximation using depth flow.
    /// Computes average absolute imbalance of depth changes over the window.
    fn compute_vpin(&self) -> f64 {
        if self.bid_depths.len() < 2 {
            return 0.0;
        }

        let n = self.bid_depths.len();
        let mut total_imbalance = 0.0;
        let mut total_volume = 0.0;

        for i in 1..n {
            let bid_delta = self.bid_depths[i] - self.bid_depths[i - 1];
            let ask_delta = self.ask_depths[i] - self.ask_depths[i - 1];

            let net_flow = bid_delta - ask_delta;
            let abs_flow = bid_delta.abs() + ask_delta.abs();

            total_imbalance += net_flow.abs();
            total_volume += abs_flow;
        }

        if total_volume > 0.0 {
            (total_imbalance / total_volume).min(1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_engine_warmup() {
        let mut engine = SignalEngine::new(20);
        // Not enough data yet
        assert!(engine.compute().is_none());

        for i in 0..5 {
            engine.push(0.5 + i as f64 * 0.01, 0.02, 0.0, 1000.0, 1000.0);
        }
        assert!(engine.compute().is_some());
    }

    #[test]
    fn test_zscore_stable_data() {
        let mut engine = SignalEngine::new(20);
        // Push constant data — z-score should be ~0
        for _ in 0..20 {
            engine.push(0.50, 0.02, 0.0, 1000.0, 1000.0);
        }
        let sigs = engine.compute().unwrap();
        assert!(sigs.price_zscore.abs() < 0.01);
    }

    #[test]
    fn test_zscore_detects_deviation() {
        let mut engine = SignalEngine::new(20);
        for _ in 0..19 {
            engine.push(0.50, 0.02, 0.0, 1000.0, 1000.0);
        }
        // Spike
        engine.push(0.60, 0.02, 0.0, 1000.0, 1000.0);
        let sigs = engine.compute().unwrap();
        assert!(sigs.price_zscore > 2.0, "Z-score should detect spike");
    }

    #[test]
    fn test_vpin_balanced_flow() {
        let mut engine = SignalEngine::new(20);
        // Truly balanced flow: both bid and ask depths move in lockstep.
        // When both sides change by the same amount, net_flow ≈ 0 → low VPIN.
        for i in 0..20 {
            let delta = if i % 2 == 0 { 50.0 } else { -50.0 };
            engine.push(0.50, 0.02, 0.0, 1000.0 + delta, 1000.0 + delta);
        }
        let sigs = engine.compute().unwrap();
        // Balanced depth changes produce near-zero net flow → low VPIN
        assert!(
            sigs.vpin < 0.1,
            "VPIN should be near zero for balanced flow, got {}",
            sigs.vpin
        );
    }

    #[test]
    fn test_vpin_toxic_flow() {
        let mut engine = SignalEngine::new(20);
        // Toxic flow: bid depth increases while ask depth decreases (aggressive buying).
        // This creates high net_flow → high VPIN.
        for i in 0..20 {
            engine.push(0.50, 0.02, 0.0, 1000.0 + (i as f64) * 20.0, 1000.0 - (i as f64) * 20.0);
        }
        let sigs = engine.compute().unwrap();
        assert!(
            sigs.vpin > 0.5,
            "VPIN should be high for toxic one-sided flow, got {}",
            sigs.vpin
        );
    }
}
